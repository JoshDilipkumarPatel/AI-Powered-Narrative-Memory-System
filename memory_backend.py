"""
Abstract backend interface with two implementations:
- InMemoryBackend (current implementation)
- ChromaBackend (paper-compliant persistent storage)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""
    
    @abstractmethod
    def add(self, memory: Dict[str, Any]) -> str:
        """Add a memory and return its ID."""
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single memory by ID."""
        pass
    
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query by embedding similarity."""
        pass
    
    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata."""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all memories (for decay processing)."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass


class InMemoryBackend(MemoryBackend):
    """Current in-memory implementation (no persistence)."""
    
    def __init__(self):
        self._store: List[Dict[str, Any]] = []
    
    def add(self, memory: Dict[str, Any]) -> str:
        self._store.append(memory)
        return memory["id"]
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for mem in self._store:
            if mem.get("id") == memory_id:
                return mem
        return None
    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return self._store[:top_k] if len(self._store) > top_k else self._store.copy()
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        for mem in self._store:
            if mem.get("id") == memory_id:
                if "metadata" in updates:
                    mem.setdefault("metadata", {}).update(updates["metadata"])
                else:
                    mem.update(updates)
                return True
        return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        return self._store.copy()
    
    def delete(self, memory_id: str) -> bool:
        for i, mem in enumerate(self._store):
            if mem.get("id") == memory_id:
                self._store.pop(i)
                return True
        return False


class ChromaBackend(MemoryBackend):
    """Paper-compliant ChromaDB backend with persistence."""
    
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "narrative_memory"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add(self, memory: Dict[str, Any]) -> str:
        """Add memory to ChromaDB."""
        mem_id = memory["id"]
        embedding = memory.get("embedding", [])
        
        metadata = {
            "content_summary": memory.get("content_summary", ""),
            "raw": memory.get("raw", "")[:1000],  # Truncate for metadata size limits
            "importance_score": float(memory.get("metadata", {}).get("importance_score", 0.5)),
            "access_count": int(memory.get("metadata", {}).get("access_count", 1)),
            "timestamp": memory.get("metadata", {}).get("timestamp", ""),
            # Store complex fields as JSON strings
            "metadata_json": json.dumps(memory.get("metadata", {}))
        }
        
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[memory.get("content_summary", "")],
            metadatas=[metadata]
        )
        return mem_id
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single memory."""
        try:
            result = self.collection.get(ids=[memory_id], include=['embeddings', 'metadatas', 'documents'])
            if result['ids']:
                return self._chroma_to_memory(result, 0)
            return None
        except Exception:
            return None
    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query by embedding similarity."""
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['embeddings', 'metadatas', 'documents', 'distances']
        )
        
        memories = []
        if result['ids'] and result['ids'][0]:
            for i in range(len(result['ids'][0])):
                mem = self._chroma_to_memory_from_query(result, i)
                # Convert distance to similarity score (assuming L2 distance)
                distance = result['distances'][0][i] if 'distances' in result else 0
                mem['score'] = 1.0 / (1.0 + distance)  # Simple conversion
                memories.append(mem)
        
        return memories
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata."""
        try:
            # ChromaDB update requires all fields
            current = self.get(memory_id)
            if not current:
                return False
            
            # Merge updates
            if "metadata" in updates:
                current.setdefault("metadata", {}).update(updates["metadata"])
            
            # Update in ChromaDB
            metadata = {
                "content_summary": current.get("content_summary", ""),
                "raw": current.get("raw", "")[:1000],
                "importance_score": float(current.get("metadata", {}).get("importance_score", 0.5)),
                "access_count": int(current.get("metadata", {}).get("access_count", 1)),
                "timestamp": current.get("metadata", {}).get("timestamp", ""),
                "metadata_json": json.dumps(current.get("metadata", {}))
            }
            
            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"ChromaBackend.update failed: {e}")
            return False
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all memories (for decay processing)."""
        result = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
        memories = []
        for i in range(len(result['ids'])):
            memories.append(self._chroma_to_memory(result, i))
        return memories
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def _chroma_to_memory(self, result: Dict, idx: int) -> Dict[str, Any]:
        """Convert ChromaDB result format to internal memory format."""
        metadata = result['metadatas'][idx]
        
        # Restore complex metadata from JSON
        try:
            metadata_dict = json.loads(metadata.get('metadata_json', '{}'))
        except Exception:
            metadata_dict = {}
        
        return {
            "id": result['ids'][idx],
            "content_summary": metadata.get('content_summary', ''),
            "raw": metadata.get('raw', ''),
            "embedding": result['embeddings'][idx] if 'embeddings' in result else [],
            "metadata": metadata_dict
        }
    
    def _chroma_to_memory_from_query(self, result: Dict, idx: int) -> Dict[str, Any]:
        """Convert ChromaDB query result format."""
        metadata = result['metadatas'][0][idx]
        
        try:
            metadata_dict = json.loads(metadata.get('metadata_json', '{}'))
        except Exception:
            metadata_dict = {}
        
        return {
            "id": result['ids'][0][idx],
            "content_summary": metadata.get('content_summary', ''),
            "raw": metadata.get('raw', ''),
            "embedding": result['embeddings'][0][idx] if 'embeddings' in result else [],
            "metadata": metadata_dict
        }


def create_backend(backend_type: str = "memory", **kwargs) -> MemoryBackend:
    """
    Factory to create backend instances.
    
    Args:
        backend_type: "memory" or "chroma"
        **kwargs: Backend-specific arguments (persist_dir for ChromaDB)
    """
    if backend_type == "chroma":
        return ChromaBackend(**kwargs)
    else:
        return InMemoryBackend()