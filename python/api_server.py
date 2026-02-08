"""Living Stream REST API Server - FastAPI-based HTTP API for AI Model Context Management."""

import sys
import os
import argparse
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

# Imports are relative to python package
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from python.llm_node import LLMNode
from python.cnn_node import CNNNode
from python.context_cache import ContextCache
from python.context import Context
from python.node_hierarchy import NodeHierarchy
from python.config import ConfigLoader, ConfigManager, load_config


# ============= Pydantic Models =============

class NodeCreateRequest(BaseModel):
    """Request model for creating a node."""
    slot: int = Field(..., ge=0, le=999, description="Slot index for the node")
    node_type: str = Field(..., pattern="^(llm|cnn)$", description="Node type: 'llm' or 'cnn'")
    model_path: str = Field(default="models/default.bin", description="Path to model file")


class NodeResponse(BaseModel):
    """Response model for node information."""
    slot: int
    node_type: str
    model_path: str
    parameter_count: Optional[int] = None
    layer_count: Optional[int] = None
    layer_sizes: List[int] = []


class ContextResponse(BaseModel):
    """Response model for context data."""
    weights_count: int
    config: Dict[str, float]
    metadata: Dict[str, str]
    model_type: str
    model_path: str
    slot_index: str


class BuildResponse(BaseModel):
    """Response model for context build operation."""
    slot: int
    context: ContextResponse


class CacheResponse(BaseModel):
    """Response model for cache operations."""
    slot: int
    cached: bool
    cache_size: int


class StatusResponse(BaseModel):
    """Response model for system status."""
    nodes_count: int
    cached_contexts: int
    cache_max_size: int
    thread_pool: str = "asyncio"
    cache_policy: str = "LRU"


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    service: str = "living-stream-api"


class CacheListResponse(BaseModel):
    """Response model for cache list."""
    slots: List[int]
    count: int


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class ParentRequest(BaseModel):
    """Request model for setting parent."""
    parent_slot: Optional[int] = Field(None, description="Parent slot index, null for root")


class GroupRequest(BaseModel):
    """Request model for group operations."""
    group_name: str = Field(..., description="Name of the group")


class HierarchyResponse(BaseModel):
    """Response model for hierarchy information."""
    slot: int
    parent: Optional[int] = None
    children: List[int] = []
    ancestors: List[int] = []
    groups: List[str] = []


class GroupListResponse(BaseModel):
    """Response model for group listing."""
    group_name: str
    members: List[int]
    count: int


class HierarchyStatsResponse(BaseModel):
    """Response model for hierarchy statistics."""
    total_nodes: int
    root_nodes: int
    trees_count: int
    groups_count: int
    avg_tree_depth: float
    max_tree_depth: int


# ============= Global State =============

_nodes: Dict[int, Union[LLMNode, CNNNode]] = {}
_cache: ContextCache = ContextCache()
_hierarchy: NodeHierarchy = NodeHierarchy()

# Config state
_config_loader: Optional[ConfigLoader] = None
_config_manager: Optional[ConfigManager] = None
_config_path: Optional[str] = None
_config_environment: Optional[str] = None
_dev_mode: bool = True


# ============= Context Serialization =============

def context_to_response(context: Context) -> ContextResponse:
    """Convert Context dataclass to response model."""
    weights = context.get_weights()
    return ContextResponse(
        weights_count=len(weights) if hasattr(weights, '__len__') else 0,
        config=context.get_config(),
        metadata=context.metadata.copy(),
        model_type=context.get_metadata("model_type"),
        model_path=context.get_metadata("model_path"),
        slot_index=context.get_metadata("slot_index")
    )


def node_to_response(slot: int, node: Union[LLMNode, CNNNode]) -> NodeResponse:
    """Convert node to response model."""
    response = NodeResponse(
        slot=slot,
        node_type="llm" if isinstance(node, LLMNode) else "cnn",
        model_path=node._model_path
    )
    if isinstance(node, LLMNode):
        response.parameter_count = node.get_parameter_count()
    elif isinstance(node, CNNNode):
        response.layer_count = node.get_layer_count()
        response.layer_sizes = node.get_layer_sizes()
    return response


# ============= Lifespan Handler =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    _nodes.clear()
    _cache.clear()
    _hierarchy.clear()
    yield
    # Shutdown
    _nodes.clear()
    _cache.clear()
    _hierarchy.clear()


# ============= FastAPI App =============

app = FastAPI(
    title="Living Stream API",
    description="REST API for AI Model Context Management - Node storage, context building, and caching",
    version="1.0.0",
    lifespan=lifespan
)


# ============= Node Management Endpoints =============

@app.post("/nodes", response_model=NodeResponse, status_code=status.HTTP_201_CREATED, tags=["Nodes"])
async def create_node(request: NodeCreateRequest) -> NodeResponse:
    """Create a new model node (LLM or CNN).
    
    Creates an LLMNode or CNNNode at the specified slot index.
    """
    slot = request.slot
    
    if slot in _nodes:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Node already exists at slot {slot}"
        )
    
    if request.node_type == "llm":
        node = LLMNode(slot, request.model_path)
    else:
        node = CNNNode(slot, request.model_path)
    
    _nodes[slot] = node
    return node_to_response(slot, node)


@app.get("/nodes/{slot}", response_model=NodeResponse, tags=["Nodes"])
async def get_node(slot: int) -> NodeResponse:
    """Get node information by slot index."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return node_to_response(slot, _nodes[slot])


@app.get("/nodes", response_model=List[NodeResponse], tags=["Nodes"])
async def list_nodes() -> List[NodeResponse]:
    """List all created nodes."""
    return [node_to_response(slot, node) for slot, node in sorted(_nodes.items())]


@app.delete("/nodes/{slot}", response_model=MessageResponse, tags=["Nodes"])
async def delete_node(slot: int) -> MessageResponse:
    """Delete a node by slot index."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    del _nodes[slot]
    return MessageResponse(message=f"Node deleted from slot {slot}")


# ============= Context Building Endpoints =============

@app.post("/nodes/{slot}/build", response_model=BuildResponse, tags=["Context"])
async def build_context_sync(slot: int) -> BuildResponse:
    """Build context synchronously for a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    context = _nodes[slot].build_context()
    return BuildResponse(slot=slot, context=context_to_response(context))


@app.post("/nodes/{slot}/build-async", response_model=BuildResponse, tags=["Context"])
async def build_context_async(slot: int) -> BuildResponse:
    """Build context asynchronously for a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    context = await _nodes[slot].async_build_context()
    return BuildResponse(slot=slot, context=context_to_response(context))


# ============= Cache Operations Endpoints =============

@app.post("/cache/{slot}", response_model=CacheResponse, tags=["Cache"])
async def cache_context(slot: int) -> CacheResponse:
    """Build and cache context for a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    context = _nodes[slot].build_context()
    _cache.cache_context(slot, context)
    
    return CacheResponse(
        slot=slot,
        cached=True,
        cache_size=_cache.size()
    )


@app.get("/cache/{slot}", response_model=BuildResponse, tags=["Cache"])
async def get_cached_context(slot: int) -> BuildResponse:
    """Retrieve cached context for a slot."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    context = _cache.get_cached_context(slot)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached context at slot {slot}"
        )
    
    return BuildResponse(slot=slot, context=context_to_response(context))


@app.delete("/cache/{slot}", response_model=MessageResponse, tags=["Cache"])
async def remove_cached_context(slot: int) -> MessageResponse:
    """Remove cached context for a slot."""
    removed = _cache.remove_context(slot)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached context at slot {slot}"
        )
    return MessageResponse(message=f"Cached context removed from slot {slot}")


@app.delete("/cache", response_model=MessageResponse, tags=["Cache"])
async def clear_cache() -> MessageResponse:
    """Clear all cached contexts."""
    _cache.clear()
    return MessageResponse(message="All cached contexts cleared")


@app.get("/cache", response_model=CacheListResponse, tags=["Cache"])
async def list_cache() -> CacheListResponse:
    """List all cached contexts."""
    return CacheListResponse(
        slots=sorted([slot for slot in range(1000) if slot in _cache]),
        count=_cache.size()
    )


# ============= System Endpoints =============

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status() -> StatusResponse:
    """Get system status."""
    return StatusResponse(
        nodes_count=len(_nodes),
        cached_contexts=_cache.size(),
        cache_max_size=_cache.get_max_size()
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


# ============= Root Endpoint =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "living-stream-api",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/status"
    }


# ============= Hierarchy Endpoints =============

@app.post("/nodes/{slot}/parent", response_model=MessageResponse, tags=["Hierarchy"])
async def set_parent(slot: int, request: ParentRequest) -> MessageResponse:
    """Set the parent of a node.
    
    Creates a parent-child relationship. Use null parent_slot to make it a root.
    """
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    if request.parent_slot is not None and request.parent_slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at parent slot {request.parent_slot}"
        )
    
    # Set parent in hierarchy manager
    success = _hierarchy.set_parent(slot, request.parent_slot)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Would create a cycle in hierarchy"
        )
    
    # Update node's parent slot
    _nodes[slot].set_parent(request.parent_slot)
    
    parent_msg = f"parent set to slot {request.parent_slot}" if request.parent_slot else "made a root node"
    return MessageResponse(message=f"Node at slot {slot} {parent_msg}")


@app.get("/nodes/{slot}/parent", response_model=Optional[int], tags=["Hierarchy"])
async def get_parent(slot: int) -> Optional[int]:
    """Get the parent of a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_parent(slot)


@app.get("/nodes/{slot}/children", response_model=List[int], tags=["Hierarchy"])
async def get_children(slot: int) -> List[int]:
    """Get all children of a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_children(slot)


@app.get("/nodes/{slot}/ancestors", response_model=List[int], tags=["Hierarchy"])
async def get_ancestors(slot: int) -> List[int]:
    """Get the ancestry path from root to the node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_ancestors(slot)


@app.get("/nodes/{slot}/descendants", response_model=List[int], tags=["Hierarchy"])
async def get_descendants(slot: int) -> List[int]:
    """Get all descendants of a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_descendants(slot)


@app.get("/nodes/{slot}/root", response_model=Optional[int], tags=["Hierarchy"])
async def get_root(slot: int) -> Optional[int]:
    """Get the root node of the hierarchy tree."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_root(slot)


@app.get("/nodes/{slot}/hierarchy", response_model=HierarchyResponse, tags=["Hierarchy"])
async def get_hierarchy(slot: int) -> HierarchyResponse:
    """Get complete hierarchy information for a node."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    return HierarchyResponse(
        slot=slot,
        parent=_hierarchy.get_parent(slot),
        children=_hierarchy.get_children(slot),
        ancestors=_hierarchy.get_ancestors(slot),
        groups=_hierarchy.get_groups(slot)
    )


# ============= Group Endpoints =============

@app.post("/nodes/{slot}/groups", response_model=MessageResponse, tags=["Groups"])
async def add_to_group(slot: int, request: GroupRequest) -> MessageResponse:
    """Add a node to a group."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    _hierarchy.add_to_group(slot, request.group_name)
    _nodes[slot].add_to_group(request.group_name)
    
    return MessageResponse(
        message=f"Node at slot {slot} added to group '{request.group_name}'"
    )


@app.delete("/nodes/{slot}/groups/{group_name}", response_model=MessageResponse, tags=["Groups"])
async def remove_from_group(slot: int, group_name: str) -> MessageResponse:
    """Remove a node from a group."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    
    removed = _hierarchy.remove_from_group(slot, group_name)
    _nodes[slot].remove_from_group(group_name)
    
    if removed:
        return MessageResponse(
            message=f"Node at slot {slot} removed from group '{group_name}'"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node not in group '{group_name}'"
        )


@app.get("/nodes/{slot}/groups", response_model=List[str], tags=["Groups"])
async def get_groups(slot: int) -> List[str]:
    """Get all groups a node belongs to."""
    if slot not in _nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found at slot {slot}"
        )
    return _hierarchy.get_groups(slot)


@app.get("/groups", response_model=List[str], tags=["Groups"])
async def list_groups() -> List[str]:
    """List all group names."""
    return _hierarchy.get_all_groups()


@app.get("/groups/{group_name}", response_model=GroupListResponse, tags=["Groups"])
async def list_group_members(group_name: str) -> GroupListResponse:
    """List all nodes in a group."""
    return GroupListResponse(
        group_name=group_name,
        members=_hierarchy.list_by_group(group_name),
        count=len(_hierarchy.list_by_group(group_name))
    )


# ============= Hierarchy Stats =============

@app.get("/hierarchy/stats", response_model=HierarchyStatsResponse, tags=["Hierarchy"])
async def get_hierarchy_stats() -> HierarchyStatsResponse:
    """Get hierarchy statistics."""
    return HierarchyStatsResponse(**_hierarchy.get_stats())


# ============= Config Endpoints =============

class ConfigStatusResponse(BaseModel):
    """Response model for config status."""
    config_loaded: bool
    config_path: Optional[str] = None
    environment: Optional[str] = None
    dev_mode: bool = True
    storage_directory: Optional[str] = None
    cache_size: Optional[int] = None
    nodes_count: int = 0


class ReloadResponse(BaseModel):
    """Response model for config reload."""
    message: str
    reloaded: bool
    nodes_loaded: int


@app.get("/config/status", response_model=ConfigStatusResponse, tags=["Config"])
async def get_config_status() -> ConfigStatusResponse:
    """Get current configuration status."""
    nodes_count = len(_nodes)
    
    storage_dir = None
    cache_size = None
    
    if _config_manager is not None:
        resolved = _config_manager.get_resolved_config()
        if resolved is not None:
            storage_dir = resolved.storage_directory
            cache_size = resolved.cache_size
    
    return ConfigStatusResponse(
        config_loaded=_config_loader is not None,
        config_path=_config_path,
        environment=_config_environment,
        dev_mode=_dev_mode,
        storage_directory=storage_dir,
        cache_size=cache_size,
        nodes_count=nodes_count
    )


@app.post("/config/reload", response_model=ReloadResponse, tags=["Config"])
async def reload_config() -> ReloadResponse:
    """Reload configuration from file (dev mode only).
    
    Only available when dev_mode is enabled.
    """
    if not _dev_mode:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Config reload is only available in dev mode"
        )
    
    if _config_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No configuration file loaded"
        )
    
    # Reload configuration
    global _config_loader, _config_manager
    
    try:
        _config_loader = ConfigLoader(_config_path)
        _config_loader.load()
        
        # Log any warnings
        for warning in _config_loader.get_warnings():
            import logging
            logging.warning(f"[Config] {warning}")
        
        _config_manager = ConfigManager(_config_loader)
        _config_manager.resolve(_config_environment)
        
        # Clear existing state
        _nodes.clear()
        _cache.clear()
        _hierarchy.clear()
        
        # Apply new config
        resolved = _config_manager.get_resolved_config()
        nodes_loaded = 0
        
        for node_config in resolved.nodes:
            slot = node_config.slot
            
            if node_config.node_type == "llm":
                _nodes[slot] = LLMNode(
                    slot,
                    node_config.config.get("model_path", "models/default.bin")
                )
            else:
                _nodes[slot] = CNNNode(
                    slot,
                    node_config.config.get("model_path", "models/default.bin")
                )
            
            # Set parent
            if node_config.parent is not None:
                _nodes[slot].set_parent(node_config.parent)
                _hierarchy.set_parent(slot, node_config.parent)
            
            # Add to groups
            for group in node_config.groups:
                _nodes[slot].add_to_group(group)
                _hierarchy.add_to_group(slot, group)
            
            # Register in hierarchy
            _hierarchy.register_node(slot)
            nodes_loaded += 1
        
        return ReloadResponse(
            message=f"Configuration reloaded from {_config_path}",
            reloaded=True,
            nodes_loaded=nodes_loaded
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload configuration: {str(e)}"
        )


def load_config_from_args():
    """Load configuration from command-line arguments."""
    global _config_loader, _config_manager, _config_path, _config_environment, _dev_mode
    
    parser = argparse.ArgumentParser(description="Living Stream API Server")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration YAML file")
    parser.add_argument("--env", "-e", type=str, default=None, help="Environment name")
    parser.add_argument("--production", action="store_true", help="Enable production mode (disables hot-reload)")
    
    args, _ = parser.parse_known_args()
    
    if args.production:
        _dev_mode = False
    
    if args.config:
        _config_path = args.config
        _config_environment = args.env
        
        try:
            _config_loader = ConfigLoader(args.config)
            _config_loader.load()
            
            # Log warnings
            for warning in _config_loader.get_warnings():
                import logging
                logging.warning(f"[Config] {warning}")
            
            _config_manager = ConfigManager(_config_loader)
            resolved = _config_manager.resolve(args.env)
            
            # Apply config to create nodes
            for node_config in resolved.nodes:
                slot = node_config.slot
                
                if node_config.node_type == "llm":
                    _nodes[slot] = LLMNode(
                        slot,
                        node_config.config.get("model_path", "models/default.bin")
                    )
                else:
                    _nodes[slot] = CNNNode(
                        slot,
                        node_config.config.get("model_path", "models/default.bin")
                    )
                
                # Set parent
                if node_config.parent is not None:
                    _nodes[slot].set_parent(node_config.parent)
                    _hierarchy.set_parent(slot, node_config.parent)
                
                # Add to groups
                for group in node_config.groups:
                    _nodes[slot].add_to_group(group)
                    _hierarchy.add_to_group(slot, group)
                
                # Register in hierarchy
                _hierarchy.register_node(slot)
            
            print(f"✓ Loaded {len(resolved.nodes)} nodes from configuration")
            
        except Exception as e:
            import logging
            logging.error(f"Failed to load configuration: {e}")


# ============= Main Entry Point =============

def main():
    """Main entry point for running the server."""
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
