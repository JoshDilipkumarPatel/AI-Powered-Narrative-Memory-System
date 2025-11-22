"""
Run this script to diagnose retrieval issues:
python debug_retrieval.py
"""

import sys
from encoder import generate_embedding
from memory_store import (
    add_memory_to_ltm, 
    retrieve_memories_from_ltm,
    _MEM_STORE,
    _FAISS_RETRIEVER,
    FAISS_AVAILABLE,
    rebuild_faiss_index
)

print("=" * 60)
print("RETRIEVAL DIAGNOSTIC TOOL")
print("=" * 60)

print("\n[1] Checking FAISS availability...")
print(f"    FAISS Available: {FAISS_AVAILABLE}")
if FAISS_AVAILABLE:
    print(f"    Index Status: {_FAISS_RETRIEVER.index is not None}")
    if _FAISS_RETRIEVER.index:
        print(f"    Index Size: {_FAISS_RETRIEVER.index.ntotal} vectors")
else:
    print("    ⚠️ FAISS not available - install with: pip install faiss-cpu")

print("\n[2] Checking memory store...")
print(f"    Total Memories: {len(_MEM_STORE)}")

if len(_MEM_STORE) == 0:
    print("    📝 No memories found. Adding test memory...")
    
    test_story = "Norman was an engineer fascinated by aerial photography. He tested a custom-built drone."
    
    test_memory = {
        "id": "test_001",
        "content_summary": test_story,
        "raw": test_story,
        "embedding": generate_embedding(test_story),
        "metadata": {
            "importance_score": 0.5,
            "access_count": 1
        }
    }
    
    mem_id = add_memory_to_ltm(test_memory)
    print(f"    ✅ Added test memory: {mem_id}")
    print(f"    Total Memories Now: {len(_MEM_STORE)}")

print("\n[3] Checking embeddings...")
for i, mem in enumerate(_MEM_STORE[:5]):
    emb = mem.get("embedding")
    print(f"    Memory {i+1}:")
    print(f"      ID: {mem.get('id')}")
    print(f"      Has Embedding: {emb is not None}")
    if emb:
        print(f"      Embedding Dimension: {len(emb)}")
    print(f"      Summary: {mem.get('content_summary', '')[:60]}...")

print("\n[4] Checking FAISS index...")
if FAISS_AVAILABLE:
    if _FAISS_RETRIEVER.index is None:
        print("    ⚠️ Index is None. Attempting rebuild...")
        if rebuild_faiss_index():
            print("    ✅ Index rebuilt successfully")
            print(f"    Index Size: {_FAISS_RETRIEVER.index.ntotal} vectors")
        else:
            print("    ❌ Index rebuild failed")
    else:
        print(f"    ✅ Index active: {_FAISS_RETRIEVER.index.ntotal} vectors")
        print(f"    Memory IDs tracked: {len(_FAISS_RETRIEVER.memory_ids)}")

print("\n[5] Testing retrieval...")
test_query = "What was Norman testing?"

print(f"    Query: '{test_query}'")

if FAISS_AVAILABLE:
    print("\n    [FAISS Retrieval]")
    results_faiss = retrieve_memories_from_ltm(test_query, use_faiss=True, top_k=3)
    print(f"    Results: {len(results_faiss)}")
    for i, r in enumerate(results_faiss[:3], 1):
        print(f"      {i}. Score: {r.get('score', 0):.3f}")
        print(f"         Semantic: {r.get('debug_semantic', 0):.3f}")
        print(f"         Lexical: {r.get('debug_lexical', 0):.3f}")
        print(f"         Summary: {r.get('content_summary', '')[:50]}...")

print("\n    [Fallback Retrieval]")
results_fallback = retrieve_memories_from_ltm(test_query, use_faiss=False, top_k=3)
print(f"    Results: {len(results_fallback)}")
for i, r in enumerate(results_fallback[:3], 1):
    print(f"      {i}. Score: {r.get('score', 0):.3f}")
    print(f"         Semantic: {r.get('debug_semantic', 0):.3f}")
    print(f"         Lexical: {r.get('debug_lexical', 0):.3f}")
    print(f"         Summary: {r.get('content_summary', '')[:50]}...")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if not FAISS_AVAILABLE:
    print("❌ Install FAISS: pip install faiss-cpu")

if len(_MEM_STORE) == 0:
    print("❌ No memories in store - add stories first")

if FAISS_AVAILABLE and _FAISS_RETRIEVER.index is None:
    print("❌ FAISS index not built - check add_memory_to_ltm()")

embeddings_missing = sum(1 for m in _MEM_STORE if not m.get("embedding"))
if embeddings_missing > 0:
    print(f"❌ {embeddings_missing} memories missing embeddings")

if len(results_fallback) == 0 and len(_MEM_STORE) > 0:
    print("❌ Retrieval returning no results - check embedding generation")

if len(results_fallback) > 0 and max(r.get('score', 0) for r in results_fallback) < 0.3:
    print("⚠️ Low confidence scores - consider adjusting min_confidence threshold")

print("\n✅ Diagnostic complete!")