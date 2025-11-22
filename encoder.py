"""
Embedding generation with proper numpy handling.
"""

_model = None

def _ensure_model():
    """Load the sentence transformer model once."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[ENCODER] Model loaded successfully")
        except Exception as e:
            print(f"[ENCODER] Failed to load model: {e}")
            raise
    return _model

def generate_embedding(text: str):
    """
    Generate embedding vector for text.
    
    Args:
        text: Input text string
    
    Returns:
        List of floats representing the embedding
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")
    
    try:
        model = _ensure_model()     
        embedding = model.encode(text)
        
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            return [float(x) for x in embedding]
            
    except Exception as e:
        print(f"[ENCODER] Error generating embedding: {e}")
        print(f"[ENCODER] Text preview: {text[:100]}...")
        raise