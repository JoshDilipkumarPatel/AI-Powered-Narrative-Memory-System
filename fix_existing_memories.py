"""
This script will:
1. Check all memories in the store
2. Generate embeddings for any that are missing them
3. Rebuild the FAISS index
4. Save the index to disk

Run: python fix_existing_memories.py
"""

from encoder import generate_embedding
from memory_store import (
    _MEM_STORE,
    rebuild_faiss_index,
    save_faiss_index,
    FAISS_AVAILABLE
)

print("=" * 60)
print("MEMORY REPAIR TOOL")
print("=" * 60)

print(f"\n📊 Current Status:")
print(f"   Total memories: {len(_MEM_STORE)}")
print(f"   FAISS available: {FAISS_AVAILABLE}")

memories_without_embeddings = []
for i, mem in enumerate(_MEM_STORE):
    if not mem.get("embedding"):
        memories_without_embeddings.append((i, mem))

print(f"   Memories missing embeddings: {len(memories_without_embeddings)}")

if len(memories_without_embeddings) == 0:
    print("\n✅ All memories have embeddings!")
else:
    print(f"\n🔧 Fixing {len(memories_without_embeddings)} memories...")
    
    for idx, (i, mem) in enumerate(memories_without_embeddings, 1):
        content = mem.get("content_summary") or mem.get("raw") or ""
        
        if content:
            print(f"   [{idx}/{len(memories_without_embeddings)}] Generating embedding for memory {mem.get('id')}")
            try:
                embedding = generate_embedding(content)
                mem["embedding"] = embedding
                print(f"       ✅ Done ({len(embedding)} dimensions)")
            except Exception as e:
                print(f"       ❌ Failed: {e}")
        else:
            print(f"   [{idx}/{len(memories_without_embeddings)}] ⚠️ Memory {mem.get('id')} has no content")

if FAISS_AVAILABLE:
    print("\n🔨 Rebuilding FAISS index...")
    if rebuild_faiss_index():
        print("   ✅ Index rebuilt successfully")
        
        print("\n💾 Saving index to disk...")
        if save_faiss_index():
            print("   ✅ Index saved")
        else:
            print("   ❌ Save failed")
    else:
        print("   ❌ Index rebuild failed")
else:
    print("\n⚠️ FAISS not available - skipping index rebuild")

print("\n" + "=" * 60)
print("FINAL STATUS")
print("=" * 60)

memories_with_embeddings = sum(1 for m in _MEM_STORE if m.get("embedding"))
print(f"✅ Memories with embeddings: {memories_with_embeddings}/{len(_MEM_STORE)}")

if FAISS_AVAILABLE:
    from memory_store import _FAISS_RETRIEVER
    if _FAISS_RETRIEVER.index:
        print(f"✅ FAISS index size: {_FAISS_RETRIEVER.index.ntotal} vectors")
    else:
        print("❌ FAISS index not built")

print("\n🎉 Repair complete! Restart your app to use the fixed memories.")