from typing import Dict
from encoder import generate_embedding
import traceback

try:
    from summarizer_abstractive import summarize_abstractive
    use_abstractive = True
except ImportError:
    from summarizer import summarize_memory
    use_abstractive = False
    print("[INGESTION] Abstractive summarizer not available, using extractive")


class LocalLLMWrapper:
    """Minimal wrapper for summarization."""
    def __call__(self, prompt_text: str):
        try:
            if use_abstractive:
                summary = summarize_abstractive(prompt_text, min_len=20, max_len=80)
            else:
                summary = summarize_memory(prompt_text, min_len=20, max_len=80)
        except Exception as e:
            print(f"[INGESTION] Summarization failed: {e}")
            summary = (prompt_text[:200] + "...") if len(prompt_text) > 200 else prompt_text
        
        import json
        fallback_json = {
            "title": (prompt_text[:60] + "...") if len(prompt_text) > 60 else prompt_text,
            "content_summary": summary,
            "characters": [],
            "setting": None,
            "key_events": [],
            "themes": [],
            "importance_score": 0.5
        }
        return json.dumps(fallback_json)


def ingest_story(raw_story_text: str, llm_chain=None) -> Dict:
    """
    Processes and prepares story data for memory ingestion.
    
    Args:
        raw_story_text: The story text to ingest
        llm_chain: Optional LLM wrapper (for compatibility)
    
    Returns:
        Dictionary with story data including embedding
    """
    if not raw_story_text or not raw_story_text.strip():
        print("[INGESTION] Empty story text provided")
        return None
    
    raw_story_text = raw_story_text.strip()
    
    if len(raw_story_text) > 150:
        content_summary = raw_story_text[:150] + "..."
    else:
        content_summary = raw_story_text
    
    print(f"[INGESTION] Generating embedding for story ({len(raw_story_text)} chars)")
    
    embedding = None
    retry_count = 0
    max_retries = 3
    
    while embedding is None and retry_count < max_retries:
        try:
            embedding = generate_embedding(raw_story_text)
            print(f"[INGESTION] ✅ Embedding generated: {len(embedding)} dimensions")
        except Exception as e:
            retry_count += 1
            print(f"[INGESTION] ❌ Embedding attempt {retry_count} failed: {e}")
            
            if retry_count >= max_retries:
                print("[INGESTION] ⚠️ All embedding attempts failed!")
                print(f"[INGESTION] Error details: {traceback.format_exc()}")
                
                print("[INGESTION] Cannot store memory without embedding - rejecting story")
                return None
    
    if embedding is None:
        print("[INGESTION] FATAL: No embedding generated after retries")
        return None
    
    parsed = {
        "id": str(hash(raw_story_text)),
        "content_summary": content_summary,
        "raw": raw_story_text,
        "embedding": embedding,  
        "metadata": {
            "importance_score": 0.5,
            "access_count": 1
        }
    }
    
    print(f"[INGESTION] ✅ Memory prepared: id={parsed['id']}, embedding_dim={len(embedding)}")
    
    return parsed