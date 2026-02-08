"""Configuration management for Living Stream.

Provides YAML configuration file loading with environment support.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from typing_extensions import TypedDict

import yaml

logger = logging.getLogger(__name__)


class NodeConfig(TypedDict, total=False):
    """Configuration for a single node."""
    slot: int
    type: str
    name: Optional[str]
    config: Dict[str, Any]
    parent: Optional[int]
    groups: List[str]


class GroupConfig(TypedDict, total=False):
    """Configuration for a group."""
    description: Optional[str]


class StorageConfig(TypedDict, total=False):
    """Storage configuration."""
    directory: str
    cache_size: int


class DefaultsConfig(TypedDict, total=False):
    """Default configuration applied to all environments."""
    storage: StorageConfig


class EnvironmentConfig(TypedDict, total=False):
    """Environment-specific configuration."""
    storage: StorageConfig


class LivingStreamConfig(TypedDict, total=False):
    """Main configuration structure."""
    version: str
    defaults: DefaultsConfig
    environments: Dict[str, EnvironmentConfig]
    nodes: List[NodeConfig]
    groups: Dict[str, GroupConfig]


@dataclass
class ResolvedNodeConfig:
    """Resolved node configuration after applying defaults and environment."""
    slot: int
    node_type: str
    name: Optional[str]
    config: Dict[str, Any]
    parent: Optional[int]
    groups: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "type": self.node_type,
            "name": self.name,
            "config": self.config,
            "parent": self.parent,
            "groups": list(self.groups)
        }


@dataclass
class ResolvedConfig:
    """Fully resolved configuration after applying environment."""
    storage_directory: str
    cache_size: int
    nodes: List[ResolvedNodeConfig]
    groups: Dict[str, str]  # name -> description


class ConfigWarning(UserWarning):
    """Warning for configuration issues."""
    pass


class ConfigLoader:
    """Load and validate Living Stream configuration files."""
    
    SUPPORTED_VERSIONS = ["1.0"]
    
    def __init__(self, config_path: str, warn_on_unknown: bool = True):
        """Initialize config loader.
        
        Args:
            config_path: Path to the YAML configuration file
            warn_on_unknown: Whether to warn on unknown config keys
        """
        self.config_path = Path(config_path)
        self.warn_on_unknown = warn_on_unknown
        self._raw_config: Optional[LivingStreamConfig] = None
        self._warnings: List[str] = []
    
    def load(self) -> LivingStreamConfig:
        """Load configuration from file.
        
        Returns:
            Raw configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        self._warnings = []
        
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Parse YAML
        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        
        if config is None:
            config = {}
        
        # Validate and warn about unknown keys
        self._validate_keys(config, "root", self._get_schema_keys())
        
        # Validate version
        version = config.get("version", "1.0")
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported config version: {version}. "
                f"Supported versions: {self.SUPPORTED_VERSIONS}"
            )
        
        self._raw_config = config
        return config
    
    def _get_schema_keys(self) -> Set[str]:
        """Get valid top-level configuration keys."""
        return {
            "version", "defaults", "environments", "nodes", "groups"
        }
    
    def _validate_keys(
        self, 
        obj: Any, 
        path: str, 
        valid_keys: Set[str],
        is_list_item: bool = False
    ) -> None:
        """Recursively validate configuration keys and warn on unknowns.
        
        Args:
            obj: Object to validate
            path: Current path in config (for error messages)
            valid_keys: Set of valid keys at this level
            is_list_item: Whether this is a list item (dict without enforced keys)
        """
        if not isinstance(obj, dict):
            return
        
        for key in obj.keys():
            if key not in valid_keys and self.warn_on_unknown:
                self._warnings.append(
                    f"Unknown config key '{path}.{key}' - ignoring"
                )
                logger.warning(f"[Config] Unknown key: {path}.{key}")
    
    def get_warnings(self) -> List[str]:
        """Get list of configuration warnings.
        
        Returns:
            List of warning messages
        """
        return self._warnings.copy()
    
    def validate(self) -> bool:
        """Validate configuration structure.
        
        Returns:
            True if valid, False otherwise
            
        Raises:
            RuntimeError: If config hasn't been loaded yet
        """
        if self._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        
        # Validate nodes
        nodes = self._raw_config.get("nodes", [])
        if not isinstance(nodes, list):
            raise ValueError("'nodes' must be a list")
        
        seen_slots: Set[int] = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ValueError(f"Node at index {i} must be a dictionary")
            
            slot = node.get("slot")
            if slot is None:
                raise ValueError(f"Node at index {i} missing required 'slot' field")
            if not isinstance(slot, int):
                raise ValueError(f"Node at index {i} 'slot' must be an integer")
            if slot in seen_slots:
                raise ValueError(f"Duplicate slot number: {slot}")
            seen_slots.add(slot)
            
            node_type = node.get("type")
            if node_type is None:
                raise ValueError(f"Node at slot {slot} missing required 'type' field")
            if node_type not in ("llm", "cnn"):
                raise ValueError(
                    f"Node at slot {slot} type must be 'llm' or 'cnn', got '{node_type}'"
                )
        
        # Validate groups (if present)
        groups = self._raw_config.get("groups", {})
        if not isinstance(groups, dict):
            raise ValueError("'groups' must be a dictionary")
        
        return True


class ConfigManager:
    """Apply configuration to create nodes and hierarchy."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize config manager.
        
        Args:
            config_loader: Loaded configuration
        """
        self.loader = config_loader
        self._resolved_config: Optional[ResolvedConfig] = None
    
    def resolve(self, environment: Optional[str] = None) -> ResolvedConfig:
        """Resolve configuration for an environment.
        
        Args:
            environment: Environment name (dev/staging/production), or None for defaults
            
        Returns:
            Resolved configuration with environment applied
        """
        if self.loader._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        
        raw = self.loader._raw_config
        
        # Start with defaults
        defaults = raw.get("defaults", {})
        env_configs = raw.get("environments", {})
        
        # Get environment config
        env_name = environment or "default"
        env_config = env_configs.get(env_name, {})
        
        # Merge storage settings (defaults < environment)
        storage = defaults.get("storage", {})
        storage.update(env_config.get("storage", {}))
        
        storage_directory = storage.get("directory", "./storage")
        cache_size = storage.get("cache_size", 1000)
        
        # Resolve nodes
        resolved_nodes: List[ResolvedNodeConfig] = []
        node_configs = raw.get("nodes", [])
        
        for node in node_configs:
            config = node.get("config", {})
            groups = set(node.get("groups", []))
            
            resolved = ResolvedNodeConfig(
                slot=node["slot"],
                node_type=node["type"],
                name=node.get("name"),
                config=config,
                parent=node.get("parent"),
                groups=groups
            )
            resolved_nodes.append(resolved)
        
        # Resolve groups
        group_descriptions: Dict[str, str] = {}
        raw_groups = raw.get("groups", {})
        for name, desc in raw_groups.items():
            group_descriptions[name] = desc.get("description", "") if isinstance(desc, dict) else ""
        
        self._resolved_config = ResolvedConfig(
            storage_directory=storage_directory,
            cache_size=cache_size,
            nodes=resolved_nodes,
            groups=group_descriptions
        )
        
        return self._resolved_config
    
    def get_resolved_config(self) -> Optional[ResolvedConfig]:
        """Get the resolved configuration.
        
        Returns:
            ResolvedConfig or None if not resolved yet
        """
        return self._resolved_config


def load_config(
    config_path: str, 
    environment: Optional[str] = None,
    validate: bool = True
) -> ResolvedConfig:
    """Convenience function to load and resolve configuration.
    
    Args:
        config_path: Path to YAML configuration file
        environment: Environment name (dev/staging/production)
        validate: Whether to validate the configuration
        
    Returns:
        Resolved configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    loader = ConfigLoader(config_path)
    raw_config = loader.load()
    
    if validate:
        loader.validate()
    
    # Log warnings
    warnings = loader.get_warnings()
    for warning in warnings:
        logger.warning(f"[Config] {warning}")
    
    manager = ConfigManager(loader)
    return manager.resolve(environment)


# ============= CLI Integration =============

def add_config_args(parser) -> None:
    """Add configuration arguments to argparse parser.
    
    Args:
        parser: ArgumentParser or add_argument Group
    """
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--env", "-e",
        type=str,
        default=None,
        help="Environment name (dev/staging/production)"
    )
    
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate configuration file without running"
    )


def apply_config_to_cli(
    args, 
    nodes_dict: Dict[int, Any], 
    hierarchy: Any,
    storage_dir: Optional[str] = None
) -> Optional[str]:
    """Apply configuration to CLI state.
    
    Args:
        args: Parsed command-line arguments
        nodes_dict: Dictionary to populate with nodes
        hierarchy: NodeHierarchy instance
        storage_dir: Output storage directory if specified
        
    Returns:
        Error message if any, None on success
    """
    if not args.config:
        return None
    
    try:
        config = load_config(args.config, args.env)
        
        if args.validate:
            print(f"✓ Configuration valid: {args.config}")
            print(f"  Environment: {args.env or 'default'}")
            print(f"  Storage: {config.storage_directory}")
            print(f"  Cache size: {config.cache_size}")
            print(f"  Nodes: {len(config.nodes)}")
            print(f"  Groups: {len(config.groups)}")
            return "VALIDATED"  # Special return to indicate validation mode
        
        if storage_dir:
            # Use storage from config
            pass  # Storage dir handling
        
        # Create nodes from config
        from python.llm_node import LLMNode
        from python.cnn_node import CNNNode
        
        for node_config in config.nodes:
            slot = node_config.slot
            
            if node_config.node_type == "llm":
                nodes_dict[slot] = LLMNode(
                    slot,
                    node_config.config.get("model_path", "models/default.bin")
                )
            else:
                nodes_dict[slot] = CNNNode(
                    slot,
                    node_config.config.get("model_path", "models/default.bin")
                )
            
            # Set parent relationship
            if node_config.parent is not None:
                nodes_dict[slot].set_parent(node_config.parent)
            
            # Add to groups
            for group in node_config.groups:
                nodes_dict[slot].add_to_group(group)
                hierarchy.add_to_group(slot, group)
            
            # Register in hierarchy
            hierarchy.register_node(slot)
            if node_config.parent is not None:
                hierarchy.set_parent(slot, node_config.parent)
        
        print(f"✓ Loaded {len(config.nodes)} nodes from configuration")
        
    except FileNotFoundError as e:
        return f"Configuration file not found: {e}"
    except ValueError as e:
        return f"Invalid configuration: {e}"
    
    return None


if __name__ == "__main__":
    # Test loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Living Stream config")
    parser.add_argument("config", nargs="?", help="Path to config file")
    add_config_args(parser)
    
    args = parser.parse_args()
    
    if args.config:
        try:
            loader = ConfigLoader(args.config)
            raw = loader.load()
            loader.validate()
            
            print("✓ Configuration valid")
            print(f"  Version: {raw.get('version', '1.0')}")
            print(f"  Nodes: {len(raw.get('nodes', []))}")
            print(f"  Groups: {len(raw.get('groups', {}))}")
            
            if loader.get_warnings():
                print("\nWarnings:")
                for w in loader.get_warnings():
                    print(f"  - {w}")
            
            # Test resolve for each environment
            manager = ConfigManager(loader)
            for env in ["default"] + list(raw.get("environments", {}).keys()):
                resolved = manager.resolve(env)
                print(f"\n✓ Resolved for '{env}':")
                print(f"  Storage: {resolved.storage_directory}")
                print(f"  Cache: {resolved.cache_size}")
                print(f"  Nodes: {len(resolved.nodes)}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python -m python.config <config.yaml>")
        sys.exit(1)
