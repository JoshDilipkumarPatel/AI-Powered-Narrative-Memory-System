"""
RAG module with local LLM generation and FAISS integration.
FIXED: Reduced confidence thresholds for better recall.
"""

import time
from typing import Dict, List, Optional

from memory_store import retrieve_memories_from_ltm, _MEM_STORE, FAISS_AVAILABLE

_GENERATOR = None

def _get_generator():
    """Lazy-load the generator model."""
    global _GENERATOR
    if _GENERATOR is None:
        try:
            from generator import LocalGenerator
            print("[RAG] Initializing LocalGenerator (this may take a moment)...")
            _GENERATOR = LocalGenerator()
            print("[RAG] Generator ready!")
        except ImportError:
            print("[RAG] Warning: generator.py not found.")
            _GENERATOR = False
        except Exception as e:
            print(f"[RAG] Failed to load generator: {e}")
            _GENERATOR = False
    return _GENERATOR if _GENERATOR is not False else None

TOP_K = 5
MAX_CONTEXT_CHARS = 1200 
MIN_CONFIDENCE = 0.25  



def _build_context_from_memories(memories: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build combined context from retrieved memories."""
    if not memories:
        return ""
    
    context_parts = []
    total_chars = 0
    
    for idx, mem in enumerate(memories, start=1):
        text = mem.get("content_summary") or mem.get("raw") or ""
        text = text.strip()
        
        if not text:
            continue
        
        passage = f"{text}\n\n"
        
        if total_chars + len(passage) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                passage = f"{text[:remaining-20]}...\n\n"
                context_parts.append(passage)
            break
        
        context_parts.append(passage)
        total_chars += len(passage)
    
    return "".join(context_parts).strip()


def _calculate_confidence(memories: List[Dict]) -> float:
    """Calculate overall confidence from retrieval scores."""
    if not memories:
        return 0.0
    
    top_score = float(memories[0].get("score", 0.0))
    
    if len(memories) > 1:
        scores = [float(m.get("score", 0.0)) for m in memories[:3]]
        avg_top3 = sum(scores) / len(scores)
        
        if avg_top3 > 0.4:
            return min(1.0, top_score * 1.1)
    
    return top_score


def _fallback_extractive_answer(query: str, memories: List[Dict]) -> str:
    """Fallback method when generator unavailable."""
    if not memories:
        return "I don't have enough information to answer that question."
    
    best = memories[0]
    snippet = best.get("content_summary") or best.get("raw") or ""
    
    if len(snippet) > 300:
        snippet = snippet[:300] + "..."
    
    confidence = float(best.get("score", 0.0))
    
    if confidence < 0.25: 
        return "I don't have enough information to answer that question."
    
    return f"Based on what I remember: {snippet}"


def generate_response(retrieval_context, stm_context=None, llm_chain=None,
                     use_faiss=True, hybrid_weight=0.7) -> str:
    """
    Main RAG pipeline with generator-based answers.
    FIXED: More permissive confidence thresholds.
    """
    start_time = time.time()
    
    if isinstance(retrieval_context, dict):
        query = retrieval_context.get("query", "") or ""
    else:
        query = str(retrieval_context or "")
    
    query = query.strip()
    if not query:
        return "I don't have enough information to answer that question."
    
    memories = retrieve_memories_from_ltm(
        query, 
        top_k=TOP_K,
        use_faiss=use_faiss and FAISS_AVAILABLE,
        hybrid_weight=hybrid_weight
    )
    
    retrieval_method = "faiss" if (use_faiss and FAISS_AVAILABLE) else "fallback"
    
    if not memories:
        elapsed = time.time() - start_time
        return (
            f"I don't have enough information to answer that question.\n\n"
            f"[debug] no_memories retrieval={retrieval_method} store_size={len(_MEM_STORE)} ({elapsed:.3f}s)"
        )
    
    confidence = _calculate_confidence(memories)
    
    context = _build_context_from_memories(memories, max_chars=MAX_CONTEXT_CHARS)
    
    top_mem = memories[0]
    top_sem = float(top_mem.get("debug_semantic", 0.0) if top_mem.get("debug_semantic") is not None else 0.0)
    top_lex = float(top_mem.get("debug_lexical", 0.0) if top_mem.get("debug_lexical") is not None else 0.0)
    
    print(f"[RAG] query='{query[:60]}' method={retrieval_method} conf={confidence:.3f} "
          f"top_sem={top_sem:.3f} top_lex={top_lex:.3f}")
    
    generator = _get_generator()
    
    if generator is None:
        print("[RAG] Generator unavailable, using extractive fallback")
        answer = _fallback_extractive_answer(query, memories)
        elapsed = time.time() - start_time
        return (
            f"{answer}\n\n"
            f"[debug] fallback_mode retrieval={retrieval_method} conf={confidence:.3f} ({elapsed:.3f}s)"
        )
    
    try:
        answer = generator.generate(context, query, confidence=confidence)
    except Exception as e:
        print(f"[RAG] Generation failed: {e}")
        answer = _fallback_extractive_answer(query, memories)
    
    elapsed = time.time() - start_time
    
    if "don't know" in answer.lower():
        return (
            f"{answer}\n\n"
            f"[debug] conf={confidence:.3f} sem={top_sem:.3f} lex={top_lex:.3f} "
            f"retrieval={retrieval_method} ({elapsed:.3f}s)"
        )
    else:
        source_ids = [m.get("id", "?")[:8] for m in memories[:2]]
        return (
            f"{answer}\n\n"
            f"[sources: {', '.join(source_ids)}]\n\n"
            f"[debug] conf={confidence:.3f} sem={top_sem:.3f} lex={top_lex:.3f} "
            f"retrieval={retrieval_method} ({elapsed:.3f}s)"
        )