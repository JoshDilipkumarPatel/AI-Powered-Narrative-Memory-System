"""
Abstractive summarization using Hugging Face transformers (paper-compliant).

Provides BART-based summarization as recommended in paper Section 3.4.
Falls back to extractive summarization if model unavailable.
"""

from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SUMMARIZER = None
_MODEL_LOADED = False


def _load_summarizer():
    """Load the BART summarizer (paper recommendation: facebook/bart-large-cnn)."""
    global _SUMMARIZER, _MODEL_LOADED
    
    if _MODEL_LOADED:
        return _SUMMARIZER
    
    try:
        from transformers import pipeline
        logger.info("Loading BART summarization model (this may take a moment)...")
        _SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  
        )
        _MODEL_LOADED = True
        logger.info("✅ BART model loaded successfully")
        return _SUMMARIZER
    
    except ImportError:
        logger.warning("transformers not installed. Run: pip install transformers torch")
        _MODEL_LOADED = True
        return None
    
    except Exception as e:
        logger.error(f"Failed to load BART model: {e}")
        _MODEL_LOADED = True
        return None


def summarize_abstractive(text: str, min_len: int = 20, max_len: int = 80) -> str:
    """
    Generate abstractive summary using BART (paper-compliant method).
    
    Args:
        text: Input text to summarize
        min_len: Minimum summary length in tokens
        max_len: Maximum summary length in tokens
    
    Returns:
        Abstractive summary string
    """
    if not text or len(text.strip()) < 10:
        return text.strip()
    
    summarizer = _load_summarizer()
    
    if summarizer is None:
        logger.warning("BART unavailable, using extractive fallback")
        from summarizer import summarize_memory
        return summarize_memory(text, min_len=min_len, max_len=max_len)
    
    try:
        if len(text) > 4000:
            text = text[:4000]
        
        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,  
            truncation=True
        )
        
        summary = result[0]['summary_text']
        logger.debug(f"Abstractive summary: {len(text)} → {len(summary)} chars")
        return summary
    
    except Exception as e:
        logger.error(f"Abstractive summarization failed: {e}")
        from summarizer import summarize_memory
        return summarize_memory(text, min_len=min_len, max_len=max_len)


def summarize_with_strategy(
    text: str,
    strategy: str = "auto",
    min_len: int = 20,
    max_len: int = 80
) -> str:
    """
    Flexible summarization with strategy selection.
    
    Args:
        text: Input text
        strategy: "abstractive", "extractive", or "auto"
                  - auto: uses abstractive if available, else extractive
        min_len: Minimum length
        max_len: Maximum length
    
    Returns:
        Summary string
    """
    if strategy == "extractive":
        from summarizer import summarize_memory
        return summarize_memory(text, min_len=min_len, max_len=max_len)
    
    elif strategy == "abstractive":
        return summarize_abstractive(text, min_len=min_len, max_len=max_len)
    
    else:
        return summarize_abstractive(text, min_len=min_len, max_len=max_len)


def warmup_abstractive_summarizer():
    """Pre-load the BART model to avoid first-call latency."""
    logger.info("Warming up abstractive summarizer...")
    summarizer = _load_summarizer()
    if summarizer:
        try:
            _ = summarizer("This is a test warmup sentence.", max_length=20, min_length=5)
            logger.info("✅ Summarizer warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")


def compare_summarization_methods(text: str) -> dict:
    """
    Generate summaries with both methods for comparison (evaluation helper).
    
    Returns dict with both summaries and ROUGE scores if available.
    """
    from summarizer import summarize_memory
    
    extractive = summarize_memory(text, min_len=20, max_len=80)
    abstractive = summarize_abstractive(text, min_len=20, max_len=80)
    
    result = {
        "extractive": extractive,
        "abstractive": abstractive,
        "extractive_len": len(extractive),
        "abstractive_len": len(abstractive)
    }
    
    try:
        from evaluate import load
        rouge = load("rouge")
        
        scores = rouge.compute(
            predictions=[abstractive],
            references=[extractive]
        )
        result["rouge_scores"] = scores
    
    except ImportError:
        logger.warning("evaluate library not installed (pip install evaluate)")
    
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
    
    return result