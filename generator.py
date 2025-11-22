"""
Bulletproof Extractive Generator with Complete Debugging
Handles all edge cases and provides detailed logging
"""

import numpy as np
from typing import List, Tuple, Optional
import re

class LocalGenerator:
    """Robust extractive generator with fallback mechanisms."""

    def __init__(self, model_name=None, max_new_tokens=None):
        """Initialize generator with cross-encoder if available."""
        print("Initializing extractive generator...")
        
        self.bi_encoder = None
        self.cross_encoder = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Bi-encoder loaded")
        except Exception as e:
            print(f"✗ Bi-encoder failed: {e}")
            raise
        
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✓ Cross-encoder loaded (better accuracy!)")
        except Exception as e:
            print(f"✗ Cross-encoder not available: {e}")
            print("  Will use bi-encoder fallback")
        
        print("✅ Generator ready!")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into complete sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences

    def _score_with_cross_encoder(self, query: str, sentences: List[str]) -> List[Tuple[float, int, str]]:
        """Score using cross-encoder (more accurate for Q&A)."""
        try:
            pairs = [(query, sent) for sent in sentences]
            scores = self.cross_encoder.predict(pairs)
            
            scored = [(float(score), idx, sent) 
                     for idx, (score, sent) in enumerate(zip(scores, sentences))]
            scored.sort(key=lambda x: x[0], reverse=True)
            
            return scored
        except Exception as e:
            print(f"[GENERATOR] Cross-encoder scoring failed: {e}")
            return []

    def _score_with_bi_encoder(self, query: str, sentences: List[str]) -> List[Tuple[float, int, str]]:
        """Score using bi-encoder cosine similarity."""
        try:
            query_emb = self.bi_encoder.encode(query, show_progress_bar=False)
            sent_embs = self.bi_encoder.encode(sentences, show_progress_bar=False)
            
            scored = []
            for idx, sent_emb in enumerate(sent_embs):
                similarity = np.dot(query_emb, sent_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(sent_emb)
                )
                scored.append((float(similarity), idx, sentences[idx]))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        except Exception as e:
            print(f"[GENERATOR] Bi-encoder scoring failed: {e}")
            return []

    def generate(self, context: str, query: str, confidence: Optional[float] = None) -> str:
        """
        Extract answer from context using semantic understanding.
        
        Args:
            context: Retrieved context text
            query: User question
            confidence: Retrieval confidence (optional)
            
        Returns:
            Extracted answer string
        """
        print(f"[GENERATOR] Called with context_len={len(context) if context else 0}, "
              f"confidence={confidence:.3f if confidence else 'None'}")
        
        if not context or not context.strip():
            print("[GENERATOR] Empty context - returning no info")
            return "I don't have enough information to answer that question."
        
        if confidence is not None and confidence < 0.25:
            print(f"[GENERATOR] Low confidence ({confidence:.3f}) - returning no info")
            return "I don't have enough information to answer that question."
        
        sentences = self._split_into_sentences(context)
        
        print(f"[GENERATOR] Split into {len(sentences)} sentences")
        
        if not sentences:
            print("[GENERATOR] No valid sentences found")
            return "I don't have enough information to answer that question."
        
        for i, sent in enumerate(sentences[:2]):
            print(f"[GENERATOR]   Sent {i}: {sent[:80]}...")
        
        if self.cross_encoder:
            scored = self._score_with_cross_encoder(query, sentences)
            method = "cross-encoder"
            threshold = 0.15  
        else:
            scored = self._score_with_bi_encoder(query, sentences)
            method = "bi-encoder"
            threshold = 0.25  
        
        if not scored:
            print("[GENERATOR] Scoring failed - returning no info")
            return "I don't have enough information to answer that question."
        
        best_score, best_idx, best_sentence = scored[0]
        
        print(f"[GENERATOR] method={method}, best_score={best_score:.3f}, "
              f"threshold={threshold:.3f}, sent_idx={best_idx}/{len(sentences)}")
        
        if best_score < threshold:
            print(f"[GENERATOR] Score {best_score:.3f} below threshold {threshold:.3f}")
            return "I don't have enough information to answer that question."
        
        question_lower = query.lower()
        is_explanatory = question_lower.startswith(('why ', 'how ', 'explain '))
        
        answer = best_sentence
        
        if is_explanatory and len(scored) > 1:
            second_score, second_idx, second_sentence = scored[1]
            if second_score > (threshold + 0.1) and abs(second_idx - best_idx) == 1:
                if second_idx > best_idx:
                    answer = best_sentence + " " + second_sentence
                else:
                    answer = second_sentence + " " + best_sentence
                print(f"[GENERATOR] Added adjacent sentence (score={second_score:.3f})")
        print(f"[GENERATOR] Returning answer (len={len(answer)})")
        
        return answer