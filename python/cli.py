#!/usr/bin/env python3
"""Living Stream CLI - Interactive AI Model Context Manager."""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional, Union
from .llm_node import LLMNode
from .cnn_node import CNNNode
from .context_cache import ContextCache
from .context import Context
from .node_hierarchy import NodeHierarchy
from .config import add_config_args, apply_config_to_cli, load_config


def print_help() -> None:
    """Print available commands."""
    print("\n=== Living Stream CLI ===")
    print("Available commands:")
    print("  create <llm|cnn> <slot> [path]  - Create a model node")
    print("  build <slot>                   - Build context synchronously")
    print("  build-async <slot>             - Build context asynchronously")
    print("  cache <slot>                   - Build and cache context")
    print("  get <slot>                     - Retrieve cached context")
    print("  remove <slot>                  - Remove from cache")
    print("  clear                          - Clear all cached contexts")
    print("  list                           - List cache contents")
    print("  info <slot>                    - Show node info")
    print("  status                         - Show system status")
    print("  help                           - Show this help")
    print("  quit                           - Exit")
    print("\nExamples:")
    print("  create llm 0                   - Create LLM at slot 0")
    print("  create cnn 1                   - Create CNN at slot 1")
    print("  cache 0                        - Build and cache slot 0")
    print("  get 0                          - Retrieve cached slot 0")


def trim(s: str) -> str:
    """Trim whitespace from string."""
    return s.strip()


def print_context_info(name: str, context: Optional[Context]) -> None:
    """Print context information."""
    if context is None:
        print("  (null context)")
        return
    
    print(f"  Type: {context.get_metadata('model_type')}")
    print(f"  Weights: {len(context.get_weights())} values")
    print(f"  Path: {context.get_metadata('model_path')}")
    
    config = context.get_config()
    print("  Config:")
    for key, value in config.items():
        print(f"    {key}: {value:.3f}")
    
    print("  Metadata:")
    print(f"    slot_index: {context.get_metadata('slot_index')}")
    layer_count = context.get_metadata('layer_count')
    if layer_count:
        print(f"    layer_count: {layer_count}")


def print_node_info(slot: int, node) -> None:
    """Print node information."""
    print(f"Node at slot {slot}:")
    if isinstance(node, LLMNode):
        print("  Type: LLM")
        print(f"  Parameter count: {node.get_parameter_count()}")
    elif isinstance(node, CNNNode):
        print("  Type: CNN")
        sizes = node.get_layer_sizes()
        print(f"  Layer count: {len(sizes)}")
        print(f"  Layer sizes: {' -> '.join(map(str, sizes))}")


def print_status(nodes: Dict[int, Union[LLMNode, CNNNode]], cache: ContextCache) -> None:
    """Print system status."""
    print("\n=== System Status ===")
    print(f"Created nodes: {len(nodes)}")
    for slot, node in nodes.items():
        print(f"  Slot {slot}: ", end="")
        if isinstance(node, LLMNode):
            print("LLM")
        elif isinstance(node, CNNNode):
            print("CNN")
    print(f"Cached contexts: {cache.size()}")
    print("Thread pool: asyncio")
    print("Cache policy: LRU")


async def run_cli() -> None:
    """Run the interactive CLI."""
    print("Living Stream - AI Model Context Manager")
    print("=========================================")
    print("Type 'help' for available commands.\n")

    # Global storage for created nodes
    nodes: Dict[int, Union[LLMNode, CNNNode]] = {}
    
    # Cache for built contexts
    cache = ContextCache()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not line:
            continue
        
        parts = line.split()
        command = parts[0].lower()
        
        if command in ("quit", "exit"):
            print("Goodbye!")
            break
        
        if command == "help":
            print_help()
            continue
        
        if command == "create":
            if len(parts) < 3:
                print("Usage: create <llm|cnn> <slot> [path]")
                continue
            
            type_str = parts[1]
            try:
                slot = int(parts[2])
            except ValueError:
                print("Invalid slot number")
                continue
            
            path = parts[3] if len(parts) > 3 else "models/default.bin"
            
            if type_str == "llm":
                nodes[slot] = LLMNode(slot, path)
                print(f"Created LLMNode at slot {slot} (path: {path})")
            elif type_str == "cnn":
                nodes[slot] = CNNNode(slot, path)
                print(f"Created CNNNode at slot {slot} (path: {path})")
            else:
                print(f"Unknown type: {type_str} (use 'llm' or 'cnn')")
            continue
        
        if command == "build":
            if len(parts) < 2:
                print("Usage: build <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            context = nodes[slot].build_context()
            print(f"Built context for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "build-async":
            if len(parts) < 2:
                print("Usage: build-async <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            print("Building context asynchronously...")
            context = await nodes[slot].async_build_context()
            print(f"Async build complete for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "cache":
            if len(parts) < 2:
                print("Usage: cache <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            context = nodes[slot].build_context()
            cache.cache_context(slot, context)
            print(f"Cached context for slot {slot}")
            continue
        
        if command == "get":
            if len(parts) < 2:
                print("Usage: get <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in cache:
                print(f"No cached context at slot {slot}. Use 'cache' first.")
                continue
            
            context = cache.get_cached_context(slot)
            print(f"Retrieved cached context for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "remove":
            if len(parts) < 2:
                print("Usage: remove <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if cache.remove_context(slot):
                print(f"Removed cached context at slot {slot}")
            else:
                print(f"No cached context at slot {slot}")
            continue
        
        if command == "clear":
            cache.clear()
            print("Cleared all cached contexts")
            continue
        
        if command == "list":
            print(f"Cached contexts ({cache.size()}):")
            for slot in range(100):
                if slot in cache:
                    print(f"  Slot {slot} (LRU)")
            continue
        
        if command == "info":
            if len(parts) < 2:
                print("Usage: info <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}")
                continue
            
            print_node_info(slot, nodes[slot])
            continue
        
        if command == "status":
            print_status(nodes, cache)
            continue
        
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Living Stream - AI Model Context Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m python.cli                    # Start interactive CLI
  python -m python.cli --config config.yaml     # Load from config file
  python -m python.cli --config config.yaml --validate  # Validate config only
  python -m python.cli --config config.yaml --env dev   # Use dev environment
        """
    )
    
    add_config_args(parser)
    
    return parser.parse_args()


async def run_cli_with_config(args) -> None:
    """Run CLI with optional config loading."""
    print("Living Stream - AI Model Context Manager")
    print("=========================================\n")
    
    # Global storage for created nodes
    nodes: Dict[int, Union[LLMNode, CNNNode]] = {}
    
    # Cache for built contexts
    cache = ContextCache()
    
    # Hierarchy manager
    hierarchy = NodeHierarchy()
    
    # Apply config if provided
    if args.config:
        result = apply_config_to_cli(args, nodes, hierarchy)
        if result == "VALIDATED":
            # Just validated, exit
            return
        elif result:
            print(f"Error: {result}")
            return
    
    if not nodes:
        print("Type 'help' for available commands.\n")
    
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not line:
            continue
        
        parts = line.split()
        command = parts[0].lower()
        
        if command in ("quit", "exit"):
            print("Goodbye!")
            break
        
        if command == "help":
            print_help()
            continue
        
        if command == "create":
            if len(parts) < 3:
                print("Usage: create <llm|cnn> <slot> [path]")
                continue
            
            type_str = parts[1]
            try:
                slot = int(parts[2])
            except ValueError:
                print("Invalid slot number")
                continue
            
            path = parts[3] if len(parts) > 3 else "models/default.bin"
            
            if type_str == "llm":
                nodes[slot] = LLMNode(slot, path)
                print(f"Created LLMNode at slot {slot} (path: {path})")
            elif type_str == "cnn":
                nodes[slot] = CNNNode(slot, path)
                print(f"Created CNNNode at slot {slot} (path: {path})")
            else:
                print(f"Unknown type: {type_str} (use 'llm' or 'cnn')")
            continue
        
        if command == "build":
            if len(parts) < 2:
                print("Usage: build <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            context = nodes[slot].build_context()
            print(f"Built context for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "build-async":
            if len(parts) < 2:
                print("Usage: build-async <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            print("Building context asynchronously...")
            context = await nodes[slot].async_build_context()
            print(f"Async build complete for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "cache":
            if len(parts) < 2:
                print("Usage: cache <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}. Use 'create' first.")
                continue
            
            context = nodes[slot].build_context()
            cache.cache_context(slot, context)
            print(f"Cached context for slot {slot}")
            continue
        
        if command == "get":
            if len(parts) < 2:
                print("Usage: get <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in cache:
                print(f"No cached context at slot {slot}. Use 'cache' first.")
                continue
            
            context = cache.get_cached_context(slot)
            print(f"Retrieved cached context for slot {slot}:")
            print_context_info("Context", context)
            continue
        
        if command == "remove":
            if len(parts) < 2:
                print("Usage: remove <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if cache.remove_context(slot):
                print(f"Removed cached context at slot {slot}")
            else:
                print(f"No cached context at slot {slot}")
            continue
        
        if command == "clear":
            cache.clear()
            print("Cleared all cached contexts")
            continue
        
        if command == "list":
            print(f"Cached contexts ({cache.size()}):")
            for slot in range(100):
                if slot in cache:
                    print(f"  Slot {slot} (LRU)")
            continue
        
        if command == "info":
            if len(parts) < 2:
                print("Usage: info <slot>")
                continue
            
            try:
                slot = int(parts[1])
            except ValueError:
                print("Invalid slot number")
                continue
            
            if slot not in nodes:
                print(f"No node at slot {slot}")
                continue
            
            print_node_info(slot, nodes[slot])
            continue
        
        if command == "status":
            print_status(nodes, cache)
            continue
        
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands.")


def main() -> None:
    """Main entry point."""
    import asyncio
    args = parse_args()
    asyncio.run(run_cli_with_config(args))


if __name__ == "__main__":
    main()
