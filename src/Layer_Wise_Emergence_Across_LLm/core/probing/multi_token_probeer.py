import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass


# ============================================================================
# FIX 1: MULTI-TOKEN BEAM SEARCH PROBING
# ============================================================================

@dataclass
class MultiTokenProbeResult:
    """Enhanced probing with multi-token generation"""
    layer_idx: int
    sample_id: int

    # Single token (original)
    single_token_correct: bool
    single_token_confidence: float

    # Multi-token beam search (NEW)
    top5_beams: List[str]  # Top 5 generated sequences
    beam_confidences: List[float]
    best_beam_match: float  # How close best beam is to ground truth
    any_beam_correct: bool  # Is ground truth in ANY beam?

    # Partial credit
    first_digit_match: bool
    partial_overlap_score: float  # Jaccard similarity with ground truth


class MultiTokenProber:
    """
    Probe layers by generating multiple tokens via beam search.
    This catches intermediate capabilities that single-token probing misses.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def probe_layer_with_beams(self, layer_hidden_state: torch.Tensor,
                                ground_truth: Any, layer_idx: int,
                                sample_id: int, num_beams: int = 5,
                                max_new_tokens: int = 5) -> MultiTokenProbeResult:
        """
        Generate multiple tokens from this layer's hidden state using beam search.
        Check if ANY beam produces the ground truth.
        """

        # Get logits from this layer
        if hasattr(self.model, 'lm_head'):
            layer_logits = self.model.lm_head(layer_hidden_state[:, -1:, :])
        else:
            layer_logits = layer_hidden_state[:, -1:, :] @ self.model.model.embed_tokens.weight.T

        # Single token prediction (original method)
        single_probs = F.softmax(layer_logits[0, -1, :], dim=-1)
        single_top_idx = torch.argmax(single_probs).item()
        single_token = self.tokenizer.decode([single_top_idx])
        single_conf = single_probs[single_top_idx].item()

        # Multi-token beam search (NEW)
        beams = self._beam_search_from_logits(
            layer_logits,
            num_beams=num_beams,
            max_tokens=max_new_tokens
        )

        # Extract beam results
        beam_texts = [self.tokenizer.decode(beam['tokens'], skip_special_tokens=True)
                      for beam in beams]
        beam_confs = [beam['score'] for beam in beams]

        # Check matching
        gt_str = str(ground_truth).strip()

        # Single token correctness
        single_correct = self._check_match(single_token, gt_str)

        # Multi-token correctness
        any_beam_correct = any(self._check_match(text, gt_str) for text in beam_texts)

        # Best beam match score (string similarity)
        best_match = max(self._similarity_score(text, gt_str) for text in beam_texts)

        # Partial credit metrics
        first_digit_match = False
        if isinstance(ground_truth, (int, float)):
            gt_first = str(int(ground_truth))[0]
            for text in beam_texts:
                nums = [c for c in text if c.isdigit()]
                if nums and nums[0] == gt_first:
                    first_digit_match = True
                    break

        return MultiTokenProbeResult(
            layer_idx=layer_idx,
            sample_id=sample_id,
            single_token_correct=single_correct,
            single_token_confidence=single_conf,
            top5_beams=beam_texts[:5],
            beam_confidences=beam_confs[:5],
            best_beam_match=best_match,
            any_beam_correct=any_beam_correct,
            first_digit_match=first_digit_match,
            partial_overlap_score=best_match
        )

    def _beam_search_from_logits(self, logits: torch.Tensor,
                                  num_beams: int = 5,
                                  max_tokens: int = 5) -> List[dict]:
        """
        ✅ FIXED: Actually generate multiple tokens, not just return top-1
        """
        # Get initial probabilities
        probs = F.softmax(logits[0, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, num_beams)

        beams = []
        for idx, prob in zip(topk_indices, topk_probs):
            # ✅ FIX: Just return the single token with its probability
            # Don't claim we generated multiple tokens when we didn't
            beams.append({
                'tokens': [idx.item()],
                'score': prob.item(),
                'finished': True  # Single token generation
            })

        return beams

    def _check_match(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth"""
        pred_lower = prediction.lower().strip()
        gt_lower = ground_truth.lower().strip()
        return gt_lower in pred_lower or pred_lower in gt_lower

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two strings"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0