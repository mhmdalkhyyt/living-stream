"""Living Stream REST API Server - FastAPI-based HTTP API for AI Model Context Management."""

import sys
import os
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

# Imports are relative to python package
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from python.llm_node import LLMNode
from python.cnn_node import CNNNode
from python.context_cache import ContextCache
from python.context import Context


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


# ============= Global State =============

_nodes: Dict[int, Union[LLMNode, CNNNode]] = {}
_cache: ContextCache = ContextCache()


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
    yield
    # Shutdown
    _nodes.clear()
    _cache.clear()


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
