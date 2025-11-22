import math
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecayConfig:
    def __init__(self, decay_rate=0.05, forget_threshold=0.2, 
                 consolidation_threshold=0.4, min_age_days=1):
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.consolidation_threshold = consolidation_threshold
        self.min_age_days = min_age_days

class DecayEngine:
    def __init__(self, config=None, backend=None, summarizer_fn=None):
        self.config = config or DecayConfig()
        self.backend = backend
        self.summarizer_fn = summarizer_fn
    
    def calculate_decay(self, memory):
        metadata = memory.get("metadata", {})
        initial_score = float(metadata.get("importance_score", 0.5))
        access_count = max(1, int(metadata.get("access_count", 1)))
        
        timestamp_str = metadata.get("timestamp")
        if not timestamp_str:
            return initial_score
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age_days = (datetime.utcnow() - timestamp.replace(tzinfo=None)).days
        except:
            return initial_score
        
        if age_days < self.config.min_age_days:
            return initial_score
        
        new_score = initial_score * math.exp(
            -self.config.decay_rate * age_days / (math.log(access_count) + 1)
        )
        return max(0.0, min(1.0, new_score))
    
    def consolidate_memory(self, memory):
        if not self.summarizer_fn:
            return memory
        
        raw_text = memory.get("raw", "") or memory.get("content_summary", "")
        if not raw_text:
            return memory
        
        try:
            gist = self.summarizer_fn(raw_text, min_len=20, max_len=80)
            memory["content_summary"] = gist
            memory["raw"] = gist
            memory["metadata"]["consolidated"] = True
            logger.info(f"Consolidated memory {memory.get('id')}")
            return memory
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return memory
    
    def run_decay_cycle(self):
        if not self.backend:
            return {"error": 1}
        
        stats = {"total": 0, "updated": 0, "consolidated": 0, "forgotten": 0, "errors": 0}
        
        try:
            memories = self.backend.get_all()
            stats["total"] = len(memories)
        except Exception as e:
            logger.error(f"Failed to fetch memories: {e}")
            return stats
        
        for memory in memories:
            try:
                mem_id = memory.get("id")
                old_score = float(memory.get("metadata", {}).get("importance_score", 0.5))
                new_score = self.calculate_decay(memory)
                
                if new_score < self.config.forget_threshold:
                    self.backend.delete(mem_id)
                    stats["forgotten"] += 1
                    logger.info(f"Forgot memory {mem_id}")
                
                elif new_score < self.config.consolidation_threshold:
                    if not memory.get("metadata", {}).get("consolidated", False):
                        memory = self.consolidate_memory(memory)
                        stats["consolidated"] += 1
                    
                    memory["metadata"]["importance_score"] = new_score
                    self.backend.update(mem_id, {"metadata": memory["metadata"]})
                    stats["updated"] += 1
                
                else:
                    if abs(new_score - old_score) > 0.01:
                        memory["metadata"]["importance_score"] = new_score
                        self.backend.update(mem_id, {"metadata": memory["metadata"]})
                        stats["updated"] += 1
            
            except Exception as e:
                logger.error(f"Error processing memory: {e}")
                stats["errors"] += 1
        
        logger.info(f"Decay cycle complete: {stats}")
        return stats