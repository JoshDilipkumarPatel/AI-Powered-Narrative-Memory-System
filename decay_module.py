from memory_store import _MEM_STORE as collection
import math
from datetime import datetime
DECAY_RATE = 0.05
FORGET_THRESHOLD = 0.2

def run_decay_cycle():
    """Fetches all memories, updates their importance score based on decay, and flags for consolidation."""

    all_memories = collection.get(include=['metadatas', 'ids'])

    updated_metadatas = []
    updated_ids = []

    for i, metadata in enumerate(all_memories['metadatas']):
        memory_id = all_memories['ids'][i]

        access_count = metadata.get('access_count', 1)
        initial_score = metadata.get('importance_score', 1.0)

        timestamp = datetime.fromisoformat(metadata['timestamp'])
        age_days = (datetime.utcnow() - timestamp).days

        new_score = initial_score * math.exp(-DECAY_RATE * age_days /
                                             (math.log(max(access_count, 1)) + 1))

        metadata['importance_score'] = new_score

        if new_score < FORGET_THRESHOLD:
            print(f"Memory {memory_id} has decayed ({new_score:.4f}) and should be consolidated.")
        else:
            updated_metadatas.append(metadata)
            updated_ids.append(memory_id)

    if updated_ids:
        collection.update(ids=updated_ids, metadatas=updated_metadatas)
        print(f"✅ {len(updated_ids)} memories updated in LTM.")
