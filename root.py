import time
from memory_store import retrieve_memories_from_ltm, _MEM_STORE
from encoder import generate_embedding

test_stories = [
    "A knight fought a dragon in the mountains.",
    "Scientists discovered a cure for cancer.",
    "Children played in the sunny park.",
    "The detective solved the mysterious case.",
    "Astronauts explored Mars surface."
] * 20

from memory_store import add_memory_to_ltm

for i, story in enumerate(test_stories):
    mem = {
        "id": f"test_{i}",
        "content_summary": story,
        "raw": story,
        "embedding": generate_embedding(story),
        "metadata": {"importance_score": 0.5, "access_count": 1}
    }
    add_memory_to_ltm(mem)

print(f"Added {len(_MEM_STORE)} memories\n")

query = "What happened with the dragon?"

start = time.time()
results_faiss = retrieve_memories_from_ltm(query, use_faiss=True)
time_faiss = (time.time() - start) * 1000

start = time.time()
results_fallback = retrieve_memories_from_ltm(query, use_faiss=False)
time_fallback = (time.time() - start) * 1000

print(f"FAISS Retrieval:")
print(f"  Time: {time_faiss:.2f}ms")
print(f"  Results: {len(results_faiss)}")
print(f"  Top score: {results_faiss[0]['score']:.3f}" if results_faiss else "  No results")

print(f"\nFallback Retrieval:")
print(f"  Time: {time_fallback:.2f}ms")
print(f"  Results: {len(results_fallback)}")
print(f"  Top score: {results_fallback[0]['score']:.3f}" if results_fallback else "  No results")

print(f"\nSpeedup: {time_fallback/time_faiss:.1f}x faster with FAISS")