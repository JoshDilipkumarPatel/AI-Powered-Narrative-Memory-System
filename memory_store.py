"""
Memory storage with FAISS-based semantic retrieval and hybrid ranking.

Key improvements:
- Fast vector similarity search using FAISS
- Persistent index storage
- Configurable confidence thresholds
- Hybrid semantic + lexical ranking
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install: pip install faiss-cpu")

from encoder import generate_embedding

_MEM_STORE: List[Dict[str, Any]] = []

class FAISSRetriever:
    """FAISS-based semantic retriever with confidence scoring."""
    
    def __init__(self, index_path: str = "./faiss_memory.index", 
                 min_confidence: float = 0.45):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to save/load FAISS index
            min_confidence: Minimum similarity score for retrieval (0.0-1.0)
        """
        self.index = None
        self.index_path = index_path
        self.min_confidence = min_confidence
        self.dimension = None
        self.memory_ids = []
        
        if not FAISS_AVAILABLE:
            print(" FAISS unavailable - falling back to basic retrieval")
    
    def build_index(self, memories: List[Dict[str, Any]]) -> bool:
        """
        Build FAISS index from current memories.
        
        Args:
            memories: List of memory objects with embeddings
        
        Returns:
            True if successful, False otherwise
        """
        if not FAISS_AVAILABLE:
            return False
        
        if not memories:
            print("No memories to index")
            return False
        
        embeddings = []
        ids = []
        
        for mem in memories:
            emb = mem.get("embedding")
            if emb and len(emb) > 0:
                embeddings.append(emb)
                ids.append(mem.get("id"))
        
        if not embeddings:
            print("No valid embeddings found")
            return False
        
        embeddings_np = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_np)
        
        self.dimension = embeddings_np.shape[1]
        self.memory_ids = ids
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        
        print(f"✅ FAISS index built: {len(embeddings)} vectors, dim={self.dimension}")
        return True
    
    def save_index(self) -> bool:
        """Save index to disk."""
        if not FAISS_AVAILABLE or self.index is None:
            return False
        
        try:
            faiss.write_index(self.index, self.index_path)
            
            metadata_path = self.index_path + ".meta"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'memory_ids': self.memory_ids,
                    'dimension': self.dimension
                }, f)
            
            print(f"✅ Index saved to {self.index_path}")
            return True
        except Exception as e:
            print(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load index from disk."""
        if not FAISS_AVAILABLE:
            return False
        
        if not os.path.exists(self.index_path):
            print(f"Index file not found: {self.index_path}")
            return False
        
        try:
            self.index = faiss.read_index(self.index_path)
            
            metadata_path = self.index_path + ".meta"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    self.memory_ids = meta.get('memory_ids', [])
                    self.dimension = meta.get('dimension')
            
            print(f"✅ Index loaded: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> tuple:
        """
        Retrieve top-k similar memories.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            (memory_ids, scores) - Lists of IDs and similarity scores
        """
        if not FAISS_AVAILABLE or self.index is None:
            return [], []
        
        query_vec = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        result_ids = []
        result_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_ids):
                score = float(scores[0][i])
                if score >= self.min_confidence:
                    result_ids.append(self.memory_ids[idx])
                    result_scores.append(score)
        
        return result_ids, result_scores


_FAISS_RETRIEVER = FAISSRetriever()

def add_memory_to_ltm(memory_obj: Dict[str, Any]) -> str:
    """
    Add memory to store and rebuild FAISS index.
    
    Args:
        memory_obj: Memory dictionary with id, embedding, content, etc.
    
    Returns:
        Memory ID
    """
    if "id" not in memory_obj:
        memory_obj["id"] = f"mem_{len(_MEM_STORE)}_{hash(str(memory_obj))}"
    
    if "metadata" not in memory_obj:
        memory_obj["metadata"] = {}
    
    if "timestamp" not in memory_obj["metadata"]:
        memory_obj["metadata"]["timestamp"] = datetime.utcnow().isoformat()
    
    if "importance_score" not in memory_obj["metadata"]:
        memory_obj["metadata"]["importance_score"] = 0.5
    
    if "access_count" not in memory_obj["metadata"]:
        memory_obj["metadata"]["access_count"] = 1
    
    if not memory_obj.get("embedding"):
        content = memory_obj.get("content_summary") or memory_obj.get("raw") or ""
        if content:
            print(f"[MEMORY_STORE] Generating embedding for memory {memory_obj['id']}")
            memory_obj["embedding"] = generate_embedding(content)
    
    _MEM_STORE.append(memory_obj)
    
    if FAISS_AVAILABLE and memory_obj.get("embedding"):
        print(f"[MEMORY_STORE] Rebuilding FAISS index ({len(_MEM_STORE)} memories)")
        _FAISS_RETRIEVER.build_index(_MEM_STORE)
    
    return memory_obj["id"]


def retrieve_memories_from_ltm(query: str, top_k: int = 5, 
                                use_faiss: bool = True,
                                hybrid_weight: float = 0.7) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using FAISS + hybrid ranking.
    
    Args:
        query: Query text
        top_k: Number of results
        use_faiss: Use FAISS retrieval (faster, more accurate)
        hybrid_weight: Weight for semantic score (0.0-1.0)
                       1.0 = pure semantic, 0.0 = pure lexical
    
    Returns:
        List of memory dictionaries with scores
    """
    if not _MEM_STORE:
        return []
    
    query_embedding = generate_embedding(query)
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    
    if use_faiss and FAISS_AVAILABLE and _FAISS_RETRIEVER.index is not None:
        memory_ids, semantic_scores = _FAISS_RETRIEVER.retrieve(query_embedding, top_k * 2)
        
        results = []
        for mem_id, sem_score in zip(memory_ids, semantic_scores):
            memory = next((m for m in _MEM_STORE if m.get("id") == mem_id), None)
            if memory:
                content = (memory.get("content_summary", "") or memory.get("raw", "")).lower()
                content_tokens = set(content.split())
                
                if len(query_tokens) > 0:
                    overlap = len(query_tokens & content_tokens)
                    lex_score = overlap / len(query_tokens)
                else:
                    lex_score = 0.0
                
                hybrid_score = (hybrid_weight * sem_score + 
                               (1 - hybrid_weight) * lex_score)
                
                memory["metadata"]["access_count"] = memory["metadata"].get("access_count", 1) + 1
                
                results.append({
                    **memory,
                    "score": hybrid_score,
                    "debug_semantic": sem_score,
                    "debug_lexical": lex_score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    else:
        return _fallback_retrieve(query, query_embedding, query_tokens, 
                                 top_k, hybrid_weight)


def _fallback_retrieve(query: str, query_embedding: List[float], 
                       query_tokens: set, top_k: int,
                       hybrid_weight: float) -> List[Dict[str, Any]]:
    """Fallback retrieval using manual cosine similarity."""
    results = []
    query_emb = np.array(query_embedding)
    
    for memory in _MEM_STORE:
        mem_emb = memory.get("embedding")
        if not mem_emb:
            continue
        
        mem_emb_np = np.array(mem_emb)
        semantic_score = float(np.dot(query_emb, mem_emb_np) / 
                              (np.linalg.norm(query_emb) * np.linalg.norm(mem_emb_np)))
        
        content = (memory.get("content_summary", "") or memory.get("raw", "")).lower()
        content_tokens = set(content.split())
        
        if len(query_tokens) > 0:
            overlap = len(query_tokens & content_tokens)
            lexical_score = overlap / len(query_tokens)
        else:
            lexical_score = 0.0
        
        hybrid_score = (hybrid_weight * semantic_score + 
                       (1 - hybrid_weight) * lexical_score)
        
        memory["metadata"]["access_count"] = memory["metadata"].get("access_count", 1) + 1
        
        results.append({
            **memory,
            "score": hybrid_score,
            "debug_semantic": semantic_score,
            "debug_lexical": lexical_score
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rebuild_faiss_index() -> bool:
    """
    Manually rebuild FAISS index from current memories.
    Useful after bulk operations or loading from disk.
    """
    if not FAISS_AVAILABLE:
        print("⚠️ FAISS not available")
        return False
    
    return _FAISS_RETRIEVER.build_index(_MEM_STORE)


def save_faiss_index() -> bool:
    """Save current FAISS index to disk."""
    return _FAISS_RETRIEVER.save_index()


def load_faiss_index() -> bool:
    """Load FAISS index from disk."""
    return _FAISS_RETRIEVER.load_index()

def initialize_memory_system():
    """Initialize memory system on startup."""
    if FAISS_AVAILABLE:
        if _FAISS_RETRIEVER.load_index():
            print("✅ Loaded existing FAISS index")
        else:
            print("📝 No existing index found - will build on first memory")


initialize_memory_system()