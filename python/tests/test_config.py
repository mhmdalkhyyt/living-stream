"""Tests for configuration management."""

import os
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import (
    ConfigLoader,
    ConfigManager,
    load_config,
    ResolvedNodeConfig,
    ResolvedConfig
)


class TestConfigLoader:
    """Tests for ConfigLoader class."""
    
    def test_load_basic_config(self):
        """Test loading a basic configuration file."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
    config:
      temperature: 0.7
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            config = loader.load()
            
            assert config["version"] == "1.0"
            assert len(config["nodes"]) == 1
            assert config["nodes"][0]["slot"] == 1
            assert config["nodes"][0]["type"] == "llm"
            
            os.unlink(f.name)
    
    def test_load_config_with_groups(self):
        """Test loading configuration with groups."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
    groups: ["production", "llm-models"]
  - slot: 2
    type: cnn
    groups: ["experiments"]
groups:
  production:
    description: "Production models"
  experiments:
    description: "Experimental models"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            config = loader.load()
            
            assert len(config["nodes"]) == 2
            assert "production" in config["nodes"][0]["groups"]
            assert "experiments" in config["nodes"][1]["groups"]
            assert "production" in config["groups"]
            
            os.unlink(f.name)
    
    def test_load_config_with_hierarchy(self):
        """Test loading configuration with parent-child hierarchy."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
    parent: null
  - slot: 2
    type: cnn
    parent: 1
  - slot: 3
    type: llm
    parent: 1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            config = loader.load()
            
            assert config["nodes"][0]["parent"] is None
            assert config["nodes"][1]["parent"] == 1
            assert config["nodes"][2]["parent"] == 1
            
            os.unlink(f.name)
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
  - slot: 2
    type: cnn
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            result = loader.validate()
            
            assert result is True
            os.unlink(f.name)
    
    def test_validate_missing_slot(self):
        """Test validation fails when slot is missing."""
        config_content = """
version: "1.0"
nodes:
  - type: llm
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            
            with pytest.raises(ValueError, match="missing required 'slot' field"):
                loader.validate()
            
            os.unlink(f.name)
    
    def test_validate_missing_type(self):
        """Test validation fails when type is missing."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            
            with pytest.raises(ValueError, match="missing required 'type' field"):
                loader.validate()
            
            os.unlink(f.name)
    
    def test_validate_invalid_type(self):
        """Test validation fails for invalid node type."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: invalid
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            
            with pytest.raises(ValueError, match="type must be 'llm' or 'cnn'"):
                loader.validate()
            
            os.unlink(f.name)
    
    def test_validate_duplicate_slots(self):
        """Test validation fails for duplicate slot numbers."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
  - slot: 1
    type: cnn
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            
            with pytest.raises(ValueError, match="Duplicate slot number"):
                loader.validate()
            
            os.unlink(f.name)
    
    def test_unknown_keys_warn(self):
        """Test that unknown keys generate warnings."""
        config_content = """
version: "1.0"
unknown_key: "value"
nodes:
  - slot: 1
    type: llm
    unknown_field: "value"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            config = loader.load()
            
            warnings = loader.get_warnings()
            assert len(warnings) >= 1
            assert any("Unknown config key" in w for w in warnings)
            
            os.unlink(f.name)
    
    def test_file_not_found(self):
        """Test error when config file doesn't exist."""
        loader = ConfigLoader("/nonexistent/path/config.yaml")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_invalid_yaml(self):
        """Test error for invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            
            loader = ConfigLoader(f.name)
            
            with pytest.raises(ValueError, match="Invalid YAML"):
                loader.load()
            
            os.unlink(f.name)
    
    def test_unsupported_version(self):
        """Test error for unsupported config version."""
        config_content = """
version: "2.0"
nodes: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            
            with pytest.raises(ValueError, match="Unsupported config version"):
                loader.load()
            
            os.unlink(f.name)


class TestConfigManager:
    """Tests for ConfigManager class."""
    
    def test_resolve_default_environment(self):
        """Test resolving configuration with default environment."""
        config_content = """
version: "1.0"
defaults:
  storage:
    directory: "./storage"
    cache_size: 1000
nodes:
  - slot: 1
    type: llm
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            loader.validate()
            
            manager = ConfigManager(loader)
            resolved = manager.resolve()
            
            assert resolved.storage_directory == "./storage"
            assert resolved.cache_size == 1000
            assert len(resolved.nodes) == 1
            
            os.unlink(f.name)
    
    def test_resolve_with_environment(self):
        """Test resolving configuration with environment override."""
        config_content = """
version: "1.0"
defaults:
  storage:
    directory: "./default_storage"
    cache_size: 500
environments:
  production:
    storage:
      directory: "./prod_storage"
      cache_size: 2000
nodes:
  - slot: 1
    type: llm
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            loader.validate()
            
            manager = ConfigManager(loader)
            resolved = manager.resolve("production")
            
            assert resolved.storage_directory == "./prod_storage"
            assert resolved.cache_size == 2000
            
            os.unlink(f.name)
    
    def test_resolve_node_config(self):
        """Test resolving node configuration."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
    name: "test-model"
    config:
      temperature: 0.7
    parent: null
    groups: ["test"]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            loader.validate()
            
            manager = ConfigManager(loader)
            resolved = manager.resolve()
            
            assert len(resolved.nodes) == 1
            node = resolved.nodes[0]
            assert node.slot == 1
            assert node.node_type == "llm"
            assert node.name == "test-model"
            assert node.config["temperature"] == 0.7
            assert node.parent is None
            assert "test" in node.groups
            
            os.unlink(f.name)
    
    def test_resolve_groups(self):
        """Test resolving group descriptions."""
        config_content = """
version: "1.0"
groups:
  group1:
    description: "First group"
  group2:
    description: "Second group"
nodes: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            loader = ConfigLoader(f.name)
            loader.load()
            loader.validate()
            
            manager = ConfigManager(loader)
            resolved = manager.resolve()
            
            assert resolved.groups["group1"] == "First group"
            assert resolved.groups["group2"] == "Second group"
            
            os.unlink(f.name)


class TestLoadConfig:
    """Tests for the convenience load_config function."""
    
    def test_load_config_convenience(self):
        """Test the load_config convenience function."""
        config_content = """
version: "1.0"
nodes:
  - slot: 1
    type: llm
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            resolved = load_config(f.name)
            
            assert isinstance(resolved, ResolvedConfig)
            assert len(resolved.nodes) == 1
            assert resolved.nodes[0].slot == 1
            
            os.unlink(f.name)
    
    def test_load_config_with_environment(self):
        """Test load_config with environment selection."""
        config_content = """
version: "1.0"
defaults:
  storage:
    directory: "./default"
environments:
  test:
    storage:
      directory: "./test_dir"
nodes: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            resolved = load_config(f.name, environment="test")
            
            assert resolved.storage_directory == "./test_dir"
            
            os.unlink(f.name)


class TestResolvedConfig:
    """Tests for ResolvedConfig and ResolvedNodeConfig dataclasses."""
    
    def test_resolved_node_to_dict(self):
        """Test converting ResolvedNodeConfig to dictionary."""
        node = ResolvedNodeConfig(
            slot=1,
            node_type="llm",
            name="test",
            config={"temp": 0.7},
            parent=None,
            groups={"group1", "group2"}
        )
        
        result = node.to_dict()
        
        assert result["slot"] == 1
        assert result["type"] == "llm"
        assert result["name"] == "test"
        assert result["config"]["temp"] == 0.7
        assert result["parent"] is None
        assert "group1" in result["groups"]
        assert "group2" in result["groups"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
