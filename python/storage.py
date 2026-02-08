"""PersistentStore - Hybrid SQLite + numpy storage for contexts.

This module provides persistent storage for Context objects using:
- SQLite for metadata, config, and indexing (fast queries)
- numpy .npy files for weight arrays (memory-mappable, fast I/O)

The design supports one-to-many reader patterns with concurrent read access
via memory-mapped files and SQLite's WAL mode.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import numpy as np

from .context import Context


class PersistentStore:
    """Persistent storage for contexts using SQLite + numpy files.
    
    Provides thread-safe read/write operations with:
    - SQLite WAL mode for concurrent reads during writes
    - Memory-mapped numpy arrays for fast weight access
    - Atomic file operations for crash safety
    
    Supports one-to-many reader patterns where multiple readers can
    access the same context concurrently.
    """
    
    # SQLite schema version for migrations
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        storage_dir: str = "storage",
        max_cache_size: int = 1000,
        enable_wal: bool = True,
        wal_checkpoint_freq: int = 100
    ):
        """Initialize persistent storage.
        
        Args:
            storage_dir: Base directory for storage (will be created)
            max_cache_size: Maximum number of contexts to cache in memory
            enable_wal: Enable SQLite WAL mode for concurrent reads
            wal_checkpoint_freq: checkpoint WAL every N writes
        """
        self._storage_dir = Path(storage_dir)
        self._slots_dir = self._storage_dir / "slots"
        self._wal_dir = self._storage_dir / "wal"
        self._db_path = self._storage_dir / "index.db"
        
        # Create directories
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._slots_dir.mkdir(parents=True, exist_ok=True)
        self._wal_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for hot contexts (thread-safe)
        self._cache: Dict[int, Tuple[Context, float]] = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = max_cache_size
        
        # Write counter for WAL checkpoint
        self._write_counter = 0
        self._wal_checkpoint_freq = wal_checkpoint_freq
        
        # Thread-local for connections
        self._local = threading.local()
        
        # Initialize database
        self._init_db(enable_wal)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn
    
    def _init_db(self, enable_wal: bool) -> None:
        """Initialize SQLite database with schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Enable WAL mode for concurrent reads
        if enable_wal:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        
        # Create contexts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                slot_id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_type TEXT,
                weight_file TEXT,
                weight_shape TEXT,
                weight_dtype TEXT,
                config_json TEXT,
                metadata_json TEXT,
                checksum TEXT,
                size_bytes INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_type ON contexts(model_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated ON contexts(updated_at)
        """)
        
        # Create metadata table for schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Set schema version
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        """, ("schema_version", str(self.SCHEMA_VERSION)))
        
        conn.commit()
    
    def _get_weight_path(self, slot_id: int) -> Path:
        """Get path to weight file for a slot."""
        return self._slots_dir / f"{slot_id:08d}.npy"
    
    def _serialize_context(self, context: Context) -> Dict[str, Any]:
        """Serialize context to storage format."""
        weights = context.get_weights()
        
        # Handle numpy arrays
        if isinstance(weights, np.ndarray):
            shape = weights.shape
            dtype = str(weights.dtype)
        elif isinstance(weights, list):
            shape = (len(weights),) if weights else (0,)
            dtype = 'float64'
        else:
            shape = (1,)
            dtype = 'float64'
        
        # Calculate checksum (simple hash for validation)
        weights_bytes = np.asarray(weights).tobytes()
        import hashlib
        checksum = hashlib.md5(weights_bytes).hexdigest()
        
        config_json = json.dumps(context.get_config())
        metadata_json = json.dumps(context.metadata)
        
        # Extract model_type from metadata for indexing
        model_type = context.metadata.get("model_type", "Unknown")
        
        return {
            'model_type': model_type,
            'weight_shape': json.dumps(shape),
            'weight_dtype': dtype,
            'config_json': config_json,
            'metadata_json': metadata_json,
            'checksum': checksum,
            'size_bytes': len(weights_bytes)
        }
    
    def _deserialize_context(
        self,
        row: sqlite3.Row,
        weights: Optional[np.ndarray] = None
    ) -> Context:
        """Deserialize context from storage format."""
        if weights is None:
            weight_path = self._get_weight_path(row['slot_id'])
            if weight_path.exists():
                weights = np.load(str(weight_path), mmap_mode='r')
            else:
                weights = []
        
        config = json.loads(row['config_json'] or '{}')
        metadata = json.loads(row['metadata_json'] or '{}')
        
        return Context(
            weights=weights,
            config=config,
            metadata=metadata
        )
    
    def _atomic_write(
        self,
        slot_id: int,
        context: Context,
        metadata: Dict[str, Any]
    ) -> None:
        """Perform atomic write of context and metadata.
        
        Uses write-to-temp + rename pattern for atomicity.
        """
        # Write weight file to temp location first
        weight_path = self._get_weight_path(slot_id)
        weights = context.get_weights()
        
        # Create temp file in wal directory
        temp_weight_path = self._wal_dir / f"temp_{slot_id}_{time.time_ns()}.npy"
        
        try:
            # Save weights to temp file
            np.save(str(temp_weight_path), np.asarray(weights))
            
            # Atomic rename to final location
            temp_weight_path.replace(weight_path)
            
            # Update database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO contexts (
                    slot_id, updated_at, model_type, weight_file,
                    weight_shape, weight_dtype, config_json,
                    metadata_json, checksum, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                slot_id,
                now,
                metadata.get('model_type', 'Unknown'),
                str(weight_path),
                metadata['weight_shape'],
                metadata['weight_dtype'],
                metadata['config_json'],
                metadata['metadata_json'],
                metadata['checksum'],
                metadata['size_bytes']
            ))
            
            conn.commit()
            
            # Periodic WAL checkpoint
            self._write_counter += 1
            if self._write_counter >= self._wal_checkpoint_freq:
                self._get_connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._write_counter = 0
                
        except Exception:
            # Clean up temp file on failure
            if temp_weight_path.exists():
                temp_weight_path.unlink()
            raise
    
    def save_context(self, slot_id: int, context: Context) -> None:
        """Save a context to persistent storage.
        
        Args:
            slot_id: Unique slot identifier
            context: Context to save
            
        Thread-safe: Multiple writers will be serialized.
        """
        metadata = self._serialize_context(context)
        
        with self._cache_lock:
            # Atomic write
            self._atomic_write(slot_id, context, metadata)
            
            # Update cache
            self._cache[slot_id] = (context, time.time())
            
            # Evict old entries if cache full
            if len(self._cache) > self._max_cache_size:
                # Remove oldest entries, at least 1
                evict_count = max(1, self._max_cache_size // 10)
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1]
                )
                for key, _ in sorted_items[:evict_count]:
                    del self._cache[key]
    
    def load_context(self, slot_id: int) -> Optional[Context]:
        """Load a context from persistent storage.
        
        Uses memory-mapped numpy files for fast weight access.
        Supports concurrent reads from multiple processes.
        
        Args:
            slot_id: Unique slot identifier
            
        Returns:
            Context if found, None otherwise.
        """
        # Check cache first
        with self._cache_lock:
            if slot_id in self._cache:
                context, _ = self._cache[slot_id]
                return context
        
        # Load from database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM contexts WHERE slot_id = ?",
            (slot_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Memory-map weights for fast access
        weight_path = self._get_weight_path(slot_id)
        try:
            weights = np.load(str(weight_path), mmap_mode='r')
        except (FileNotFoundError, ValueError):
            weights = None
        
        context = self._deserialize_context(row, weights)
        
        # Update cache
        with self._cache_lock:
            self._cache[slot_id] = (context, time.time())
            if len(self._cache) > self._max_cache_size:
                self._evict_old_cache()
        
        return context
    
    def _evict_old_cache(self) -> None:
        """Evict oldest cache entries."""
        evict_count = max(1, self._max_cache_size // 10)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1][1]
        )
        for key, _ in sorted_items[:evict_count]:
            del self._cache[key]
    
    def get_metadata(self, slot_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a slot without loading weights.
        
        Fast operation suitable for listing/browsing.
        
        Args:
            slot_id: Unique slot identifier
            
        Returns:
            Metadata dict or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM contexts WHERE slot_id = ?",
            (slot_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            'slot_id': row['slot_id'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'model_type': row['model_type'],
            'weight_shape': json.loads(row['weight_shape'] or '[]'),
            'weight_dtype': row['weight_dtype'],
            'config': json.loads(row['config_json'] or '{}'),
            'size_bytes': row['size_bytes'],
            'checksum': row['checksum']
        }
    
    def list_slots(
        self,
        model_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List available slots with metadata.
        
        Args:
            model_type: Filter by model type (e.g., 'LLM', 'CNN')
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of slot metadata dicts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM contexts"
        params = []
        
        if model_type:
            query += " WHERE model_type = ?"
            params.append(model_type)
        
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'slot_id': row['slot_id'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'model_type': row['model_type'],
                'weight_shape': json.loads(row['weight_shape'] or '[]'),
                'size_bytes': row['size_bytes']
            })
        
        return results
    
    def delete_context(self, slot_id: int) -> bool:
        """Delete a context from storage.
        
        Args:
            slot_id: Unique slot identifier
            
        Returns:
            True if deleted, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete from database
        cursor.execute(
            "DELETE FROM contexts WHERE slot_id = ?",
            (slot_id,)
        )
        
        if cursor.rowcount == 0:
            return False
        
        conn.commit()
        
        # Delete weight file
        weight_path = self._get_weight_path(slot_id)
        if weight_path.exists():
            weight_path.unlink()
        
        # Remove from cache
        with self._cache_lock:
            self._cache.pop(slot_id, None)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dict with storage stats.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count
        cursor.execute("SELECT COUNT(*) FROM contexts")
        total_count = cursor.fetchone()[0]
        
        # Total size
        cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM contexts")
        total_size = cursor.fetchone()[0]
        
        # By model type
        cursor.execute("""
            SELECT model_type, COUNT(*) as count
            FROM contexts
            GROUP BY model_type
        """)
        by_type = {row['model_type']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_contexts': total_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'by_model_type': by_type,
            'cached_count': len(self._cache),
            'cache_max_size': self._max_cache_size,
            'storage_dir': str(self._storage_dir),
            'db_path': str(self._db_path)
        }
    
    def exists(self, slot_id: int) -> bool:
        """Check if a slot exists in storage.
        
        Args:
            slot_id: Unique slot identifier
            
        Returns:
            True if exists, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM contexts WHERE slot_id = ?",
            (slot_id,)
        )
        
        return cursor.fetchone() is not None
    
    def clear(self) -> None:
        """Clear all stored contexts.
        
        WARNING: This deletes all data!
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM contexts")
        conn.commit()
        
        # Clear weight files
        for f in self._slots_dir.glob("*.npy"):
            f.unlink()
        
        # Clear cache
        with self._cache_lock:
            self._cache.clear()
    
    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        conn = self._get_connection()
        conn.execute("VACUUM")
    
    def close(self) -> None:
        """Close all connections and cleanup."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __enter__(self) -> 'PersistentStore':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PersistentStore("
            f"contexts={stats['total_contexts']}, "
            f"size={stats['total_size_mb']:.1f}MB, "
            f"cached={stats['cached_count']}"
            f")"
        )
