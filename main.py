"""
Emergence Archaeology: Systematic Layer-wise Capability Localization

This research framework maps the exact layer and neuron locations where specific
capabilities (math, coding, theory of mind) crystallize across model scales.

Key Components:
1. Layer-wise Probing: Test capability at each layer
2. Activation Patching: Identify causal layers
3. Causal Interventions: Verify necessity/sufficiency
4. Comprehensive Logging: Track all experiments for ablation studies
"""
"""
Emergence Archaeology: Systematic Layer-wise Capability Localization
...
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
torch._dynamo.config.ignore_logger_methods = ["warning_once", "warning", "info", "debug"]

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import re
from tqdm import tqdm
import warnings
from transformers import BitsAndBytesConfig
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings('ignore')


"""
CRITICAL IMPROVEMENTS TO EMERGENCE DETECTION

Add these three classes to your existing code to fix detection blind spots.
These will help distinguish real emergence from measurement artifacts.
"""

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


# ============================================================================
# FIX 2: LINEAR PROBING ON HIDDEN STATES
# ============================================================================

class LinearLayerProber:
    """
    Train linear classifiers on hidden states to detect if information
    is ENCODED (even if not EXPRESSED in next-token predictions).

    This is crucial: if middle layers encode the answer but don't express it,
    your current method will miss it.
    """

    def __init__(self):
        self.probes = {}  # layer_idx -> trained probe
        self.scalers = {}  # layer_idx -> StandardScaler

    def train_probe_for_layer(self, layer_idx: int,
                              hidden_states: np.ndarray,
                              labels: np.ndarray,
                              task_type: str = "classification"):
        """
        Train a linear probe on hidden states from this layer.

        Args:
            layer_idx: Which layer
            hidden_states: Shape (num_samples, hidden_dim)
            labels: Ground truth answers (num_samples,)
            task_type: "classification" or "regression"
        """

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(hidden_states)

        # Train probe
        if task_type == "classification":
            probe = LogisticRegression(max_iter=1000, random_state=42)
        else:
            from sklearn.linear_model import Ridge
            probe = Ridge(alpha=1.0)

        probe.fit(X, labels)

        # Store
        self.probes[layer_idx] = probe
        self.scalers[layer_idx] = scaler

        # Evaluate
        train_acc = probe.score(X, labels)
        return train_acc


    def compare_encoding_vs_expression_v2(self, probing_df: pd.DataFrame) -> dict:
        """
        CORRECTED VERSION with better error handling
        """
        comparison = {}

        for layer_idx in sorted(probing_df['layer_idx'].unique()):
            layer_data = probing_df[probing_df['layer_idx'] == layer_idx]

            # Expression accuracy
            expression_acc = layer_data['is_correct'].mean()

            # ✅ FIX: Better data validation
            hidden_states = []
            binary_labels = []

            for _, row in layer_data.iterrows():
                # Check if layer_hidden exists and is valid
                if 'layer_hidden' in row and row['layer_hidden'] is not None:
                    hidden_state = row['layer_hidden']

                    # ✅ FIX: Handle different data types
                    if isinstance(hidden_state, np.ndarray):
                        if hidden_state.ndim > 1:
                            hidden_state = hidden_state.flatten()
                        hidden_states.append(hidden_state)
                        binary_labels.append(row['is_correct'])
                    elif isinstance(hidden_state, (list, tuple)):
                        hidden_states.append(np.array(hidden_state).flatten())
                        binary_labels.append(row['is_correct'])

            # ✅ FIX: More robust minimum sample check
            encoding_acc = None
            if len(hidden_states) >= 10:  # Need at least 10 samples
                try:
                    X = np.array(hidden_states)
                    y = np.array(binary_labels, dtype=bool)

                    # ✅ FIX: Check for class balance
                    if y.sum() > 0 and (~y).sum() > 0:  # Both classes present
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # ✅ FIX: Use cross-validation for better estimates
                        from sklearn.model_selection import cross_val_score
                        probe = LogisticRegression(max_iter=2000, random_state=42)

                        # If enough samples, use CV; otherwise use simple fit
                        if len(X) >= 20:
                            scores = cross_val_score(probe, X_scaled, y, cv=min(5, len(X)//4))
                            encoding_acc = scores.mean()
                        else:
                            probe.fit(X_scaled, y)
                            encoding_acc = probe.score(X_scaled, y)
                    else:
                        print(f"   ⚠️ Layer {layer_idx}: Skipped - imbalanced classes")
                except Exception as e:
                    print(f"   ❌ Layer {layer_idx}: Linear probe failed - {e}")
                    encoding_acc = None
            else:
                print(f"   ⚠️ Layer {layer_idx}: Skipped - only {len(hidden_states)} samples")

            comparison[int(layer_idx)] = {
                'expression_accuracy': float(expression_acc),
                'encoding_accuracy': float(encoding_acc) if encoding_acc is not None else None,
                'gap': float(encoding_acc - expression_acc) if encoding_acc is not None else None,
                'n_samples': len(hidden_states)  # ✅ ADD: Track sample count
            }

        return comparison


    def probe_test_sample(self, layer_idx: int,
                          hidden_state: np.ndarray) -> Tuple[Any, float]:
        """
        Use trained probe to predict from hidden state.

        Returns:
            prediction: Predicted class/value
            confidence: Probability (for classification)
        """
        if layer_idx not in self.probes:
            return None, 0.0

        probe = self.probes[layer_idx]
        scaler = self.scalers[layer_idx]

        # Reshape if needed
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        # Transform and predict
        X = scaler.transform(hidden_state)
        prediction = probe.predict(X)[0]

        # Get confidence if classifier
        if hasattr(probe, 'predict_proba'):
            proba = probe.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0  # For regression

        return prediction, confidence

    def compare_encoding_vs_expression(self, probing_results_df) -> dict:
      """
      Compare when information is ENCODED (probe detects it) vs.
      EXPRESSED (next-token prediction shows it).

      This is THE KEY ANALYSIS to distinguish real emergence from artifact.
      """

      comparison = {}

      for layer_idx in sorted(probing_results_df['layer_idx'].unique()):
          layer_data = probing_results_df[probing_results_df['layer_idx'] == layer_idx]

          # Expression: Your current method (next-token accuracy)
          expression_acc = layer_data['is_correct'].mean()

          # Encoding: Linear probe accuracy (if we have it)
          if layer_idx in self.probes:
              hidden_states = np.stack(layer_data['layer_hidden'].values)
              ground_truths = layer_data['ground_truth'].values

              predictions = []
              for hs in hidden_states:
                  pred, _ = self.probe_test_sample(layer_idx, hs)
                  predictions.append(pred)

              encoding_acc = np.mean(predictions == ground_truths)
          else:
              encoding_acc = None

          # ✅ FIX: Convert numpy int64 to native Python int for JSON keys
          comparison[int(layer_idx)] = {
              'expression_accuracy': float(expression_acc),  # Also convert to float
              'encoding_accuracy': float(encoding_acc) if encoding_acc is not None else None,
              'gap': float(encoding_acc - expression_acc) if encoding_acc is not None else None
          }

      return comparison


# ============================================================================
# ENHANCED ATTENTION PATTERN ANALYSIS
# ============================================================================

@dataclass
class AttentionAnalysisResult:
    """Results from attention analysis for one sample"""
    layer_idx: int
    sample_id: int

    # Overall metrics
    math_attention_score: float  # How much attention to math tokens
    reasoning_attention_score: float  # Attention to intermediate reasoning

    # Head-level analysis
    specialized_heads: List[int]  # Which heads focus on math
    head_specialization_scores: np.ndarray  # Score per head

    # Token-level analysis
    math_token_indices: List[int]  # Which tokens are math-relevant
    attention_matrix: np.ndarray  # Full attention pattern (for visualization)


class EnhancedAttentionAnalyzer:
    """
    Analyze attention patterns to detect mathematical reasoning circuits.

    Key insights:
    1. Math-specialized heads: Attend heavily to numbers and operators
    2. Reasoning heads: Attend to previous computational steps
    3. Output heads: Aggregate information in final layers
    """

    def __init__(self, model, tokenizer, debug=False):
        self.model = model
        self.tokenizer = tokenizer
        self.debug = debug

    def analyze_sample(self, question: str, ground_truth: Any,
                      sample_id: int, task_type: str = "math") -> List[AttentionAnalysisResult]:
        """
        Analyze attention patterns for one sample across all layers.
        """
        try:
            # Format prompt
            if task_type == "math":
                prompt = f"Solve: {question}\nAnswer:"
            else:
                prompt = question

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                  padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get tokens for analysis
            token_ids = inputs['input_ids'][0].cpu().numpy()
            tokens = [self.tokenizer.decode([t]) for t in token_ids]

            if self.debug:
                print(f"\n🔍 ATTENTION ANALYSIS - Sample {sample_id}")
                print(f"Tokens: {tokens[:20]}...")  # First 20 tokens

            # Identify math-relevant tokens
            math_token_indices = self._identify_math_tokens(tokens, task_type)

            if self.debug:
                print(f"Math tokens at indices: {math_token_indices}")

            # Get attention weights - with error handling
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=False
                )

            # ✅ FIX: Check if attentions are returned
            if outputs.attentions is None:
                if self.debug:
                    print(f"⚠️ No attention weights returned for sample {sample_id}")
                return self._create_empty_attention_results(sample_id)

            attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

            results = []

            # Analyze each layer
            for layer_idx, layer_attn in enumerate(attentions):
                # layer_attn shape: (1, num_heads, seq_len, seq_len)
                if layer_attn is None:
                    continue

                layer_attn = layer_attn[0]  # Remove batch dim: (heads, seq, seq)

                # Focus on last token's attention (what does output attend to?)
                last_token_attn = layer_attn[:, -1, :]  # (num_heads, seq_len)

                # Compute metrics
                math_score = self._compute_math_attention_score(
                    last_token_attn, math_token_indices
                )

                reasoning_score = self._compute_reasoning_attention_score(
                    last_token_attn, math_token_indices, len(tokens)
                )

                # Find specialized heads
                specialized_heads, head_scores = self._find_specialized_heads(
                    last_token_attn, math_token_indices
                )

                # Store full attention matrix for this layer (for visualization)
                attention_matrix = layer_attn.cpu().numpy()  # (heads, seq, seq)

                result = AttentionAnalysisResult(
                    layer_idx=layer_idx,
                    sample_id=sample_id,
                    math_attention_score=math_score,
                    reasoning_attention_score=reasoning_score,
                    specialized_heads=specialized_heads,
                    head_specialization_scores=head_scores,
                    math_token_indices=math_token_indices,
                    attention_matrix=attention_matrix
                )

                results.append(result)

                if self.debug and layer_idx % 5 == 0:
                    print(f"Layer {layer_idx:2d}: Math={math_score:.3f}, "
                          f"Reasoning={reasoning_score:.3f}, "
                          f"Specialized heads={len(specialized_heads)}")

            return results

        except Exception as e:
            if self.debug:
                print(f"❌ Attention analysis failed for sample {sample_id}: {e}")


    def _identify_math_tokens(self, tokens: List[str], task_type: str) -> List[int]:
        """
        Identify which tokens are mathematically relevant.

        Returns list of token indices.
        """
        math_indices = []

        for idx, token in enumerate(tokens):
            token_lower = token.lower().strip()

            # Numbers
            if any(c.isdigit() for c in token):
                math_indices.append(idx)

            # Math operators
            elif token in ['+', '-', '*', '/', '=', '>', '<', '(', ')']:
                math_indices.append(idx)

            # Math words
            elif token_lower in ['sum', 'total', 'plus', 'minus', 'times',
                                'divide', 'equals', 'answer', 'result']:
                math_indices.append(idx)

            # Question words (for reasoning)
            elif token_lower in ['how', 'many', 'much', 'what', 'calculate']:
                math_indices.append(idx)

        return math_indices

    def _compute_math_attention_score(self, last_token_attn: torch.Tensor,
                                      math_indices: List[int]) -> float:
        """
        Compute how much the final token attends to math-relevant tokens.

        Args:
            last_token_attn: (num_heads, seq_len) attention weights
            math_indices: Indices of math tokens

        Returns:
            Average attention to math tokens across all heads
        """
        if not math_indices:
            return 0.0

        # Average attention to math tokens across all heads
        math_attention = last_token_attn[:, math_indices].mean().item()

        return float(math_attention)

    def _compute_reasoning_attention_score(self, last_token_attn: torch.Tensor,
                                          math_indices: List[int],
                                          seq_len: int) -> float:
        """
        Compute attention to intermediate positions (reasoning steps).

        Hypothesis: Middle tokens contain intermediate computation.
        """
        if seq_len < 10:
            return 0.0

        # Define "middle" tokens (exclude first 3 and last 3)
        middle_start = 3
        middle_end = seq_len - 3

        if middle_end <= middle_start:
            return 0.0

        middle_indices = list(range(middle_start, middle_end))

        # Remove math tokens from middle (we want non-math reasoning tokens)
        reasoning_indices = [i for i in middle_indices if i not in math_indices]

        if not reasoning_indices:
            return 0.0

        reasoning_attention = last_token_attn[:, reasoning_indices].mean().item()

        return float(reasoning_attention)

    def _find_specialized_heads(self, last_token_attn: torch.Tensor,
                               math_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Find which attention heads specialize in mathematical reasoning.

        A head is "specialized" if it attends much more to math tokens
        than the average head.

        Returns:
            specialized_heads: List of head indices
            head_scores: Score for each head
        """
        if not math_indices:
            return [], np.array([])

        num_heads = last_token_attn.shape[0]
        head_scores = np.zeros(num_heads)

        # Compute each head's attention to math tokens
        for head_idx in range(num_heads):
            head_attn = last_token_attn[head_idx, :]
            math_attn = head_attn[math_indices].mean().item()
            head_scores[head_idx] = math_attn

        # Find heads that are > 2 standard deviations above mean
        mean_score = head_scores.mean()
        std_score = head_scores.std()

        threshold = mean_score + 1.5 * std_score  # 1.5 std above mean

        specialized_heads = [i for i in range(num_heads)
                            if head_scores[i] > threshold]

        return specialized_heads, head_scores

    def aggregate_results(self, all_results: List[List[AttentionAnalysisResult]]) -> Dict:
        """
        Aggregate attention results across all samples.

        Args:
            all_results: List of [results for each sample]

        Returns:
            Dictionary with aggregated statistics
        """
        if not all_results:
            return {}

        num_layers = len(all_results[0])
        num_samples = len(all_results)

        aggregated = {
            'layer_stats': [],
            'head_specialization_by_layer': {},
            'emergence_layer': -1
        }

        # Aggregate by layer
        for layer_idx in range(num_layers):
            layer_results = [sample_results[layer_idx] for sample_results in all_results]

            # Average metrics
            avg_math_attn = np.mean([r.math_attention_score for r in layer_results])
            avg_reasoning_attn = np.mean([r.reasoning_attention_score for r in layer_results])

            # Count specialized heads across samples
            all_specialized_heads = []
            for r in layer_results:
                all_specialized_heads.extend(r.specialized_heads)

            # Find consistently specialized heads (appear in >50% of samples)
            from collections import Counter
            head_counts = Counter(all_specialized_heads)
            consistent_heads = [head for head, count in head_counts.items()
                              if count >= num_samples * 0.5]

            layer_stat = {
                'layer': layer_idx,
                'avg_math_attention': float(avg_math_attn),
                'avg_reasoning_attention': float(avg_reasoning_attn),
                'num_consistent_specialized_heads': len(consistent_heads),
                'consistent_specialized_heads': consistent_heads
            }

            aggregated['layer_stats'].append(layer_stat)
            aggregated['head_specialization_by_layer'][layer_idx] = consistent_heads

        # ✅ IMPROVED: Dynamic threshold scaling based on model size
        if hasattr(self.model, 'config'):
            model_num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            model_num_layers = len(self.model.model.layers)
        else:
            model_num_layers = num_layers

        print(f"🔍 Model has {model_num_layers} layers (analyzed {num_layers} layers)")

        # ✅ AUTOMATIC THRESHOLD SCALING
        # Larger models need higher thresholds as attention gets more diffuse
        if model_num_layers <= 16:  # 1B models (16 layers)
            threshold = 0.015  # 2%
        elif model_num_layers <= 28:  # 3B models (28 layers)
            threshold = 0.02  # 2.5%
        elif model_num_layers <= 40:  # 8B models (32-40 layers)
            threshold = 0.025  # 3%
        elif model_num_layers <= 60:  # 13B models (40-60 layers)
            threshold = 0.035  # 3.5%
        elif model_num_layers <= 90:  # 70B models (80-90 layers)
            threshold = 0.04  # 4%
        else:  # 405B models (126+ layers)
            threshold = 0.05  # 5%

        print(f"🔍 Attention emergence detection: Using threshold={threshold:.1%} for {model_num_layers}-layer model")

        # Find "attention emergence layer" (first layer with strong math attention)
        layer_stats_filtered = []
        for layer_stat in aggregated['layer_stats']:
            # Skip first 3 layers (they're just random initialization)
            if layer_stat['layer'] < 3:
                layer_stat['avg_math_attention'] = 0.0
                layer_stat['num_consistent_specialized_heads'] = 0
                layer_stat['consistent_specialized_heads'] = []
            layer_stats_filtered.append(layer_stat)

        aggregated['layer_stats'] = layer_stats_filtered

        # Re-find emergence layer after filtering
        for layer_stat in aggregated['layer_stats']:
            if layer_stat['layer'] >= 3:  # Only consider layer 3+
                if layer_stat['avg_math_attention'] > threshold:
                    aggregated['emergence_layer'] = layer_stat['layer']
                    break

        # If no layer meets threshold, find the layer with maximum math attention
        if aggregated['emergence_layer'] == -1:
            max_attention_layer = max(aggregated['layer_stats'],
                                    key=lambda x: x['avg_math_attention'])
            aggregated['emergence_layer'] = max_attention_layer['layer']
            print(f"⚠️ No layer exceeded threshold, using max attention layer {max_attention_layer['layer']} "
                  f"(math attention={max_attention_layer['avg_math_attention']:.1%})")

        return aggregated







@dataclass
class ProbeResult:
    """Results from probing a single layer"""
    # Fields WITHOUT default values (must come first)
    layer_idx: int
    model_name: str
    task_type: str
    dataset_name: str
    sample_id: int
    question: str
    ground_truth: Any
    layer_logits: np.ndarray
    layer_prediction: str
    layer_confidence: float
    is_correct: bool
    activation_mean: float
    activation_std: float
    activation_max: float
    activation_sparsity: float
    input_length: int
    output_length: int
    timestamp: str

    # Fields WITH default values (must come after)
    layer_hidden: np.ndarray = None
    multi_token_correct: bool = False
    best_beam_match: float = 0.0
    first_digit_match: bool = False
    top_beam: str = ""

@dataclass
class PatchingResult:
    """Results from activation patching experiment"""
    layer_idx: int
    model_name: str
    dataset_name: str
    sample_id: int
    question: str
    ground_truth: Any

    # Patching details
    source_run: str  # "correct" or "incorrect"
    target_run: str
    patched_prediction: str
    patched_correct: bool

    # Causal metrics
    causal_effect: float  # Change in correctness probability
    intervention_type: str  # "activation_patch", "zero_ablate", etc.

    # Activation changes
    activation_delta_mean: float
    activation_delta_max: float



class MultiTaskValidator:
    """Validate answers across different task types"""

    def __init__(self, debug=False):
        self.debug = debug
        self.math_validator = MathValidator(debug)

    def extract_and_validate(self, text: str, ground_truth: Any, task_type: str) -> Tuple[bool, Any]:
        """Extract and validate answer based on task type"""
        if task_type == "math":
            predicted = self.math_validator.extract_answer(text)
            is_correct = self.math_validator.validate(predicted, ground_truth)
            return is_correct, predicted

        elif task_type == "reasoning":  # BoolQ
            text_lower = text.lower()
            # Extract yes/no/true/false
            if any(word in text_lower for word in ['yes', 'true', 'correct']):
                predicted = True
            elif any(word in text_lower for word in ['no', 'false', 'incorrect']):
                predicted = False
            else:
                predicted = None

            is_correct = (predicted == ground_truth)
            return is_correct, predicted

        elif task_type in ["commonsense", "multiple_choice"]:
            # ✅ IMPROVED: Handle both letter and text matching
            text_lower = text.lower()
            ground_truth_str = str(ground_truth).strip()
            
            # Strategy 1: Check if ground truth is a single letter (A-E)
            if len(ground_truth_str) == 1 and ground_truth_str.upper() in 'ABCDE':
                # Ground truth is a letter - extract model's choice
                predicted_letter = self.extract_choice(text)
                if predicted_letter:
                    is_correct = predicted_letter.upper() == ground_truth_str.upper()
                    return is_correct, predicted_letter
                
                # Fallback: check if letter appears in text
                is_correct = ground_truth_str.lower() in text_lower
                return is_correct, ground_truth_str
            
            # Strategy 2: Ground truth is full text (e.g., CommonsenseQA)
            else:
                ground_truth_lower = ground_truth_str.lower()
                
                # Check containment
                is_correct = ground_truth_lower in text_lower
                
                # Also try extracting choice
                predicted = self.extract_choice(text)
                
                return is_correct, predicted

        else:
            # Default: string matching
            return str(ground_truth).lower() in text.lower(), None

    def extract_choice(self, text: str) -> Optional[str]:
        """Extract multiple choice selection (improved for HellaSwag)"""
        # Look for patterns like "Answer: A", "4.", "Option B", etc.
        patterns = [
            r'answer\s*[:\-]\s*([A-E1-5])',  # "Answer: A" or "Answer: 4"
            r'option\s*([A-E1-5])',           # "Option A"
            r'choice\s*([A-E1-5])',           # "Choice 4"
            r'^([A-E1-5])[\.:]',              # "A." or "4:"
            r'\b([A-E])\b',                   # Standalone letter
            r'\b([1-5])\b',                   # Standalone number
        ]
    
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                choice = matches[0].strip().upper()
                
                # Convert number to letter if needed (1→A, 2→B, etc.)
                if choice.isdigit():
                    num = int(choice)
                    if 1 <= num <= 5:
                        choice = chr(64 + num)  # 1→A, 2→B, 3→C, 4→D, 5→E
                
                return choice
        
        return None



class MathValidator:
    """Extract and validate mathematical answers"""

    def __init__(self, debug=False):
        self.debug = debug

    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from model output"""
        text_lower = text.lower()

        # Strategy 1: "Final Answer: X" format (highest priority)
        patterns = [
            r'final\s+answer\s*[:\-]\s*(?:.*?)(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'answer\s*[:\-]\s*(?:.*?)(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'####\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'\\boxed\{(\d+(?:,\d{3})*(?:\.\d+)?)\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    return float(matches[-1].replace(',', ''))
                except:
                    continue

        # Strategy 2: Last number in final sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            final_text = '. '.join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]
            matches = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', final_text)
            if matches:
                try:
                    return float(matches[-1].replace(',', ''))
                except:
                    pass

        return None

    def validate(self, predicted: Optional[float], ground_truth: Any, tolerance: float = 0.01) -> bool:
        """Check if prediction matches ground truth with type handling"""
        if predicted is None:
            return False

        # Handle different ground truth types
        if isinstance(ground_truth, (int, float)):
            # Numerical comparison
            return abs(predicted - ground_truth) <= tolerance
        elif isinstance(ground_truth, bool):
            # Boolean comparison - convert predicted to boolean logic
            # If predicted is close to 1, treat as True; close to 0, treat as False
            if predicted > 0.5:
                return ground_truth == True
            else:
                return ground_truth == False
        elif isinstance(ground_truth, str):
            # String comparison - convert predicted to string and check containment
            predicted_str = str(predicted)
            return predicted_str in ground_truth or ground_truth in predicted_str
        else:
            # Fallback: string representation matching
            return str(predicted) == str(ground_truth)
class SharedModelManager:
    def __init__(self, model_name: str, device: str = "cuda", debug: bool = False):
        self.model_name = model_name
        self.device = device
        self.debug = debug
        
        print(f"🔧 Loading model WITHOUT quantization: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ✅ REMOVE quantization - load full precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # ❌ NO quantization_config
            device_map="auto",
            torch_dtype=torch.float16,  # Use half precision to save memory
            output_hidden_states=True,
            output_attentions=True
        )
        self.model.eval()
        
        self.num_layers = len(self.model.model.layers)
        print(f"✅ Model loaded WITHOUT quantization: {self.num_layers} layers")
    
    def get_model(self):
        """Return the shared model instance"""
        return self.model
    
    def get_tokenizer(self):
        """Return the shared tokenizer"""
        return self.tokenizer
    
    def cleanup(self):
        """Free GPU memory when done"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            print("🗑️ Model freed from GPU")

class LayerWiseProber:
    """Probe model capabilities at each layer"""

    def __init__(self, shared_manager:SharedModelManager, debug: bool = False):
        self.model_name = shared_manager.model_name
        self.device = shared_manager.device
        self.debug = debug

        self.model = shared_manager.get_model()
        self.tokenizer = shared_manager.get_tokenizer()
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_layers = len(self.model.model.layers)
        self.validator = MultiTaskValidator(debug=debug)
        self.multi_prober = MultiTokenProber(self.model, self.tokenizer, self.device)

        # ✅ FIXED: Use EnhancedAttentionAnalyzer with model layer count
        self.attention_analyzer = EnhancedAttentionAnalyzer(
            self.model, self.tokenizer, debug=debug
        )
        # Store the actual model layer count in the analyzer
        self.attention_analyzer.model_num_layers = self.num_layers

        self.linear_prober = LinearLayerProber()
        print(f"✅ Model loaded: {self.num_layers} layers")


    
    
    def train_linear_probes(self):
        """Train linear classifiers with CORRECT binary classification"""
        if not hasattr(self, '_linear_probing_data') or not self._linear_probing_data['hidden_states']:
            print("❌ No data collected for linear probing")
            return

        self.linear_prober = LinearLayerProber()

        print(f"\n🔬 TRAINING LINEAR PROBES - BINARY CLASSIFICATION")

        probe_results = {}
        for layer_idx in sorted(self._linear_probing_data['hidden_states'].keys()):
            hidden_states = self._linear_probing_data['hidden_states'][layer_idx]
            labels = self._linear_probing_data['labels'][layer_idx]

            if len(hidden_states) < 5:
                print(f"   Layer {layer_idx}: Skipped - only {len(hidden_states)} samples")
                continue

            # ✅ CRITICAL FIX: Convert to binary classification
            # Use the SINGLE-TOKEN correctness as binary labels
            X = np.array(hidden_states)

            # Get single-token correctness for this layer from the main results
            layer_correctness = []
            for sample_id in self._linear_probing_data['sample_ids'][layer_idx]:
                # Find if this sample was correct at this layer in single-token probing
                sample_data = self.probing_results_df[
                    (self.probing_results_df['sample_id'] == sample_id) &
                    (self.probing_results_df['layer_idx'] == layer_idx)
                ]
                if len(sample_data) > 0:
                    layer_correctness.append(sample_data['is_correct'].iloc[0])
                else:
                    layer_correctness.append(False)

            y_binary = np.array(layer_correctness, dtype=bool)

            print(f"   Layer {layer_idx}: X.shape={X.shape}, y_binary: {y_binary.sum()}/{len(y_binary)} correct")

            try:
                accuracy = self.linear_prober.train_probe_for_layer(
                    layer_idx=layer_idx,
                    hidden_states=X,
                    labels=y_binary,  # Use binary correctness labels
                    task_type="classification"
                )
                probe_results[layer_idx] = accuracy
                print(f"   Layer {layer_idx:2d}: Linear probe accuracy = {accuracy:.2%}")

            except Exception as e:
                print(f"   Layer {layer_idx:2d}: Linear probe failed - {e}")

        return probe_results

    def format_prompt(self, question: str, task_type: str = "math") -> str:
        """Format question based on task type"""
        if task_type == "math":
            system_msg = "You are a mathematical problem solver. Solve the problem step by step and provide your final answer clearly marked as 'Final Answer: X'."
            return f"{system_msg}\n\nQuestion: {question}\n\nSolution:"

        elif task_type == "reasoning":  # BoolQ
            system_msg = "You are a reasoning assistant. Read the passage and answer the question with a simple 'Yes' or 'No'."
            return f"{system_msg}\n\n{question}\n\nAnswer:"

        elif task_type == "commonsense":  # Commonsense QA, HellaSwag
            system_msg = "You are a commonsense reasoning assistant. Choose the most appropriate answer from the given options and provide it clearly."
            # Extract options and format properly
            if "Options:" in question:
                parts = question.split("Options:")
                main_question = parts[0].strip()
                options = parts[1].strip()
                return f"{system_msg}\n\nQuestion: {main_question}\n\nOptions: {options}\n\nAnswer:"
            else:
                return f"{system_msg}\n\n{question}\n\nAnswer:"

        else:
            return question



    def validate_early_layer_sanity(self, layer_idx: int, top_tokens: List[str]) -> bool:
      """
      Sanity check: Early layers should NOT show capability detection
      unless there's VERY strong evidence
      """
      early_threshold = self.num_layers // 3

      if layer_idx < early_threshold:
          # List of common early-layer tokens that cause false positives
          false_positive_indicators = [
              '.', ',', 'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that',
              'for', 'it', 'with', 'as', 'was', 'on', 'but', 'not', 'this',
              'are', 'be', 'have', 'from', 'or', 'by', 'at', 'an', 'they',
              'which', 'one', 'all', 'there', 'so', 'if', 'out', 'up', 'what',
              'who', 'when', 'where', 'why', 'how', 'some', 'more', 'very',
              'just', 'then', 'than', 'also', 'well', 'only', 'now', 'such'
          ]

          # If top tokens are mostly common words, likely false positive
          common_word_count = sum(1 for token in top_tokens[:10]
                                if token.lower() in false_positive_indicators)
          if common_word_count >= 5:  # More than half are common words
              return False

      return True


    def generate_full_answer(self, question: str, task_type: str = "math", ground_truth: Any = None) -> Tuple[Optional[Any], str]:
        """Generate full answer and extract result based on task type"""
        prompt = self.format_prompt(question, task_type)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the newly generated part (after the prompt)
        prompt_length = len(prompt)
        new_generation = generated_text[prompt_length:].strip()

        if self.debug:
            print(f"📝 Full Generation:\n{generated_text}\n")
            print(f"🎯 New Generation (after prompt):\n{new_generation}\n")

        # Use appropriate extraction based on task type
        if task_type == "math":
            extracted_answer = self.validator.math_validator.extract_answer(generated_text)  # This will work now
        elif task_type == "reasoning":
            # For BoolQ, extract yes/no
            extracted_answer = self.validator.extract_and_validate(new_generation, ground_truth, task_type)[1]
        elif task_type == "commonsense":
            # For multiple choice, extract the chosen option
            extracted_answer = self.validator.extract_choice(new_generation)
            if extracted_answer is None:
                # Fallback: return the first few words of the generation
                extracted_answer = ' '.join(new_generation.split()[:10])
        else:
            extracted_answer = new_generation

        return extracted_answer, generated_text



    def probe_all_layers(self, question: str, ground_truth: Any,
                        sample_id: int, task_type: str = "math",
                        dataset_name: str = "unknown") -> List[ProbeResult]:
        """
        Probe capability at every layer by extracting hidden states
        and decoding predictions from each layer
        """
        prompt = self.format_prompt(question, task_type)

        if self.debug:
            print(f"\n{'='*80}")
            print(f"🔬 PROBING SAMPLE #{sample_id}")
            print(f"{'='*80}")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Prompt: {prompt[:200]}...")
            print(f"{'='*80}\n")

        # Get full correct answer first
        final_prediction, full_generation = self.generate_full_answer(question, task_type, ground_truth)
        final_correct = self.validator.extract_and_validate(full_generation, ground_truth, task_type)[0]

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]

        results = []

        # Get hidden states for all layers
        with torch.no_grad():
            forward_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )

            hidden_states = forward_outputs.hidden_states

        if self.debug:
            print(f"📝 Full Generation:\n{full_generation}\n")
            print(f"🎯 Final Prediction: {final_prediction} (Correct: {final_correct})")
            print(f"🔍 Probing {len(hidden_states)} layers...\n")

        # Store hidden states for linear probing
        all_layer_hidden_states = {}

        # Probe each layer
        for layer_idx in range(len(hidden_states)):
            layer_hidden = hidden_states[layer_idx]
            last_hidden = layer_hidden[:, -1, :]

            # ✅ STORE FOR LINEAR PROBING
            if layer_idx not in all_layer_hidden_states:
                all_layer_hidden_states[layer_idx] = []
            all_layer_hidden_states[layer_idx].append(last_hidden[0].detach().cpu().numpy())

            # Project to vocabulary logits
            if hasattr(self.model, 'lm_head'):
                layer_logits = self.model.lm_head(last_hidden)
            else:
                layer_logits = last_hidden @ self.model.model.embed_tokens.weight.T

            # ✅ FIX: Add .detach().cpu() before converting to numpy
            layer_logits_np = layer_logits[0].detach().cpu().numpy()

            # Get top prediction at this layer
            # Get top prediction at this layer
            top_token_id = torch.argmax(layer_logits, dim=-1).item()
            top_token = self.tokenizer.decode([top_token_id])

            layer_correct, layer_confidence = self.compute_layer_correctness(
                    layer_logits, question, ground_truth, layer_idx, task_type  # ADD THIS
                )

            if layer_idx == 19 and self.debug:
                # Get top 10 tokens for debugging
                topk_probs, topk_indices = torch.topk(F.softmax(layer_logits[0], dim=-1), 10)
                topk_tokens = [self.tokenizer.decode([idx.item()]) for idx in topk_indices]
                
                print(f"\n🔍 Layer 19 Debug - Sample {sample_id}")
                print(f"  Question: {question[:100]}...")
                print(f"  Top token: '{top_token}'")
                print(f"  Top 10 tokens: {topk_tokens}")
                print(f"  Top 10 probs: {[f'{p:.3f}' for p in topk_probs.tolist()]}")
                print(f"  Ground truth: {ground_truth}")
                print(f"  Detected as correct: {layer_correct}")
                print(f"  Confidence: {layer_confidence:.4f}")

            # ✅ FIX: Add .detach().cpu() for activations too
            layer_activations = layer_hidden[0, -1, :].detach().cpu().numpy()
            activation_mean = float(np.mean(layer_activations))
            activation_std = float(np.std(layer_activations))
            activation_max = float(np.max(np.abs(layer_activations)))
            activation_sparsity = float(np.mean(np.abs(layer_activations) < 0.01))

            # ✅ MULTI-TOKEN PROBING (NEW)
            # ✅ MULTI-TOKEN PROBING (NEW)
            multi_token_correct = False
            best_beam_match = 0.0
            first_digit_match = False
            top_beam = ""
            

            try:
                multi_result = self.multi_prober.probe_layer_with_beams(
                    layer_hidden_state=layer_hidden,
                    ground_truth=ground_truth,
                    layer_idx=layer_idx,
                    sample_id=sample_id
                )

                # Extract multi-token results
                multi_token_correct = multi_result.any_beam_correct
                best_beam_match = multi_result.best_beam_match
                first_digit_match = multi_result.first_digit_match
                top_beam = multi_result.top5_beams[0] if multi_result.top5_beams else ""

                if self.debug and multi_token_correct and not layer_correct:
                    print(f"🎯 Layer {layer_idx}: Multi-token found what single-token missed!")
                    print(f"   Beams: {multi_result.top5_beams[:3]}")

            except Exception as e:
                if self.debug:
                    print(f"⚠️ Multi-token probing failed at layer {layer_idx}: {e}")
                # Fallback values
                multi_token_correct = False
                best_beam_match = 0.0
                first_digit_match = False
                top_beam = ""

            # Store result
            result = ProbeResult(
                layer_idx=layer_idx,
                model_name=self.model_name,
                task_type=task_type,
                sample_id=sample_id,
                dataset_name=dataset_name,
                question=question,
                ground_truth=ground_truth,
                layer_logits=layer_logits_np,
                layer_prediction=top_token,
                layer_hidden=last_hidden[0].detach().cpu().numpy(),
                layer_confidence=layer_confidence,
                is_correct=layer_correct,
                activation_mean=activation_mean,
                activation_std=activation_std,
                activation_max=activation_max,
                activation_sparsity=activation_sparsity,
                input_length=input_length,
                output_length=0,
                timestamp=pd.Timestamp.now().isoformat(),
                # ✅ NEW MULTI-TOKEN FIELDS
                multi_token_correct=multi_token_correct,
                best_beam_match=best_beam_match,
                first_digit_match=first_digit_match,
                top_beam=top_beam
            )
            results.append(result)

            if self.debug and layer_idx % 5 == 0:
                multi_indicator = " 🎯" if multi_token_correct else ""
                print(f"Layer {layer_idx:2d}: Token='{top_token}' "
                    f"(correct={layer_correct}, multi={multi_token_correct}{multi_indicator}, "
                    f"conf={layer_confidence:.3f})")

        # ✅ STORE FOR LATER LINEAR PROBING TRAINING
        self._store_for_linear_probing(all_layer_hidden_states, ground_truth, sample_id)
        self._store_for_linear_probing(all_layer_hidden_states, ground_truth, sample_id)

        # ✅ NEW: ATTENTION ANALYSIS
        if not hasattr(self, '_attention_results'):
            self._attention_results = []

        try:
            if self.debug:
                print(f"\n🔍 Running attention analysis for sample {sample_id}...")

            attention_results = self.attention_analyzer.analyze_sample(
                question=question,
                ground_truth=ground_truth,
                sample_id=sample_id,
                task_type=task_type
            )

            self._attention_results.append(attention_results)

            if self.debug:
                print(f"✅ Attention analysis complete: {len(attention_results)} layers analyzed")

        except Exception as e:
            print(f"⚠️ Attention analysis failed for sample {sample_id}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()



        return results

    def _store_for_linear_probing(self, hidden_states: dict, ground_truth: Any, sample_id: int):
        """FINAL FIX: Store ONE hidden state per layer per sample for linear probing"""
        if not hasattr(self, '_linear_probing_data'):
            self._linear_probing_data = {
                'hidden_states': {},  # layer_idx -> list of SINGLE hidden states
                'labels': {},         # layer_idx -> list of labels
                'sample_ids': {}      # layer_idx -> list of sample_ids
            }

        # ✅ FINAL FIX: Store ONE hidden state per layer per sample
        for layer_idx, states_list in hidden_states.items():
            if layer_idx not in self._linear_probing_data['hidden_states']:
                self._linear_probing_data['hidden_states'][layer_idx] = []
                self._linear_probing_data['labels'][layer_idx] = []
                self._linear_probing_data['sample_ids'][layer_idx] = []

            # ✅ CRITICAL: states_list contains ALL hidden states from this sample
            # We only want the FINAL one for linear probing
            if isinstance(states_list, list) and len(states_list) > 0:
                # Take the last hidden state from this layer for this sample
                final_hidden_state = states_list[-1]  # This should be a 1D array
            else:
                final_hidden_state = states_list

            # ✅ Ensure we're storing a 1D array (flatten if needed)
            if isinstance(final_hidden_state, np.ndarray) and final_hidden_state.ndim > 1:
                final_hidden_state = final_hidden_state.flatten()

            self._linear_probing_data['hidden_states'][layer_idx].append(final_hidden_state)
            self._linear_probing_data['labels'][layer_idx].append(ground_truth)
            self._linear_probing_data['sample_ids'][layer_idx].append(sample_id)


    def compute_layer_correctness(self, layer_logits, question, ground_truth, layer_idx, task_type='math'): 
        """
        ✅ CORRECTED VERSION v4: First-high-accuracy detection
        
        Detects emergence at the FIRST layer with >70% accuracy,
        not the final stabilization layer. This catches TRUE emergence.
        """
        
        probs = F.softmax(layer_logits.detach(), dim=-1)[0]
        
        # Get top 100 predictions
        topk_probs, topk_indices = torch.topk(probs, min(100, len(probs)))
        topk_probs_np = topk_probs.detach().cpu().numpy()
        topk_indices_np = topk_indices.detach().cpu().numpy()
        topk_tokens = [self.tokenizer.decode([idx]).strip().lower() for idx in topk_indices_np]
        
        confidence = float(topk_probs_np[0])
        
        # Convert ground truth to searchable format
        if isinstance(ground_truth, (int, float)):
            gt_str = str(int(ground_truth)) if ground_truth == int(ground_truth) else str(ground_truth)
            gt_digits = set(c for c in gt_str if c.isdigit())
        elif isinstance(ground_truth, bool):
            gt_str = "yes" if ground_truth else "no"
            gt_digits = set()
        else:
            gt_str = str(ground_truth).lower()
            gt_digits = set()
        
        # ========== SIMPLIFIED DETECTION: FIRST HIGH ACCURACY ==========
        
        # Skip very early layers (first 10% or first 3 layers)
        skip_threshold = max(3, self.num_layers // 10)
        
        if layer_idx < skip_threshold:
            # Early layers: only exact matches
            for i in range(min(10, len(topk_tokens))):
                token = topk_tokens[i]
                if gt_str == token or (len(gt_str) > 2 and gt_str in token):
                    return True, float(topk_probs_np[i])
            return False, confidence
        
        # ========== MAIN DETECTION (Layer 3+) ==========
        
        # Strategy 1: Exact/substring match (highest priority)
        for i in range(min(50, len(topk_tokens))):
            token = topk_tokens[i]
            if gt_str in token or token in gt_str:
                return True, float(topk_probs_np[i])
        
        # Strategy 2: First layer with >70% accuracy gets FULL CREDIT
        # This is the KEY CHANGE - we detect emergence here, not at stabilization
        if layer_idx >= skip_threshold:
            # Check for math engagement tokens
            math_indicators = {
                'calculate', 'answer', 'final', 'result', 'solution',
                'sum', 'total', 'equals', 'multiply', 'divide', 'subtract',
                'first', 'second', 'step', 'next', 'then',
                'let', "let's", 'so', 'therefore'
            }
            
            for i in range(min(20, len(topk_tokens))):
                token = topk_tokens[i]
                if any(indicator in token for indicator in math_indicators):
                    if task_type == 'math':
                        return True, float(topk_probs_np[i])  # FULL credit
        
        # Strategy 3: Digit matching (for numerical answers)
        if isinstance(ground_truth, (int, float)) and gt_digits:
            for i in range(min(80, len(topk_tokens))):
                token = topk_tokens[i]
                token_digits = set(c for c in token if c.isdigit())
                
                if token_digits:
                    overlap = token_digits & gt_digits
                    overlap_ratio = len(overlap) / len(gt_digits) if gt_digits else 0
                    
                    if overlap_ratio >= 0.5:  # At least 50% digit match
                        return True, float(topk_probs_np[i])
        
        # Strategy 4: Boolean matching
        if isinstance(ground_truth, bool):
            yes_tokens = ['yes', 'true', 'correct', 'right']
            no_tokens = ['no', 'false', 'incorrect', 'wrong']
            target_tokens = yes_tokens if ground_truth else no_tokens
            
            for i in range(min(30, len(topk_tokens))):
                if topk_tokens[i] in target_tokens:
                    return True, float(topk_probs_np[i])
        
        return False, confidence



def calculate_patch_layers(num_layers: int, num_patches: int = 5) -> List[int]:
    """
    ✅ AUTOMATIC PATCHING LAYER SELECTION
    
    Intelligently select which layers to patch based on model size.
    Returns evenly-spaced layers covering the full depth.
    
    Args:
        num_layers: Total number of layers in the model
        num_patches: How many layers to patch (default: 5)
    
    Returns:
        List of layer indices to patch
    """
    if num_layers <= 10:
        # Very small model - patch every other layer
        return list(range(0, num_layers, 2))
    
    # Calculate spacing for even distribution
    # Always include: first layer (0), last layer (num_layers-1), and middle layers
    if num_patches >= num_layers:
        # Patch all layers
        return list(range(num_layers))
    
    # Evenly space the patches
    spacing = num_layers / (num_patches - 1)
    patch_layers = [round(spacing * i) for i in range(num_patches - 1)]
    patch_layers.append(num_layers - 1)  # Always include last layer
    
    # Remove duplicates and sort
    patch_layers = sorted(list(set(patch_layers)))
    
    print(f"📍 Auto-selected {len(patch_layers)} patch layers for {num_layers}-layer model: {patch_layers}")
    
    return patch_layers


class ActivationPatcher:
    """Perform activation patching - USES SHARED MODEL"""
    
    def __init__(self, shared_manager: SharedModelManager, debug: bool = False):
        self.model_name = shared_manager.model_name
        self.device = shared_manager.device
        self.debug = debug

        self.model = shared_manager.get_model()
        self.tokenizer = shared_manager.get_tokenizer()
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_layers = len(self.model.model.layers)
        self.validator = MultiTaskValidator(debug=debug)  # CHANGE THIS LINE TOO
        self.validator = MathValidator(debug=debug)
        print(f"✅ Patcher ready: {self.num_layers} layers")

    def patch_layer_activation(self, question: str, ground_truth: Any,
                           sample_id: int, layer_to_patch: int,
                           intervention_type: str = "residual_patch",
                           dataset_name: str = "unknown") -> PatchingResult:
        """
        Patch a specific layer's activations and measure causal effect
        """
        if self.debug:
            print(f"\n🔬 PATCHING Layer {layer_to_patch} - {intervention_type}")
            print(f"Question: {question}")

        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get baseline (unpatched) performance first
        baseline_answer, baseline_correct = self.get_baseline_performance(question, ground_truth)

        # Hook to intervene on layer
        activations_cache = {}

        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store original
            activations_cache['original'] = hidden_states.clone()

            # ✅ IMPROVED INTERVENTIONS:
            # ✅ IMPROVED INTERVENTIONS: More targeted and causal
            if intervention_type == "residual_patch":
                # More aggressive patching for better causal detection
                patch_ratio = 0.5  # Increased from 0.3
                patch_mask = torch.rand_like(hidden_states) < patch_ratio
                # Use mean instead of zero to be less destructive but still effective
                patch_value = hidden_states.mean() * torch.ones_like(hidden_states)
                hidden_states = torch.where(patch_mask, patch_value, hidden_states)

            elif intervention_type == "neuron_ablate":
                # NEW: Ablate top neurons (more targeted)
                topk_vals, topk_indices = torch.topk(hidden_states,
                                                  k=hidden_states.shape[-1] // 10,  # Top 10%
                                                  dim=-1)
                threshold = topk_vals[:, :, -1:]
                hidden_states = torch.where(hidden_states >= threshold,
                                          torch.zeros_like(hidden_states),
                                          hidden_states)

            elif intervention_type == "activation_scale":
                # Scale down activations
                hidden_states = hidden_states * 0.3

            elif intervention_type == "zero_ablate":
                # Original zero ablation (most destructive)
                hidden_states = torch.zeros_like(hidden_states)

            elif intervention_type == "mean_ablate":
                # Replace with mean activation
                mean_activation = hidden_states.mean(dim=[0, 1], keepdim=True)
                hidden_states = mean_activation.expand_as(hidden_states)

            elif intervention_type == "topk_ablate":
                # Only keep top-k activations
                k = hidden_states.shape[-1] // 4  # Keep top 25%
                topk_vals, _ = torch.topk(hidden_states, k, dim=-1)
                threshold = topk_vals[:, :, -1:]
                hidden_states = torch.where(hidden_states >= threshold,
                                          hidden_states,
                                          torch.zeros_like(hidden_states))

            activations_cache['intervened'] = hidden_states.clone()

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        # Register hook
        target_layer = self.model.model.layers[layer_to_patch]
        handle = target_layer.register_forward_hook(intervention_hook)

        # Generate with intervention
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.1  # Add slight temperature for stability
                )
        except Exception as e:
            if self.debug:
                print(f"⚠️ Generation error: {e}")
            # Fallback: use forward pass instead of generate
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Extract logits and manually generate token
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                outputs = torch.cat([inputs['input_ids'], next_token.unsqueeze(0)], dim=1)

        handle.remove()

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        patched_prediction = self.validator.extract_answer(generated_text)
        patched_correct = self.validator.validate(patched_prediction, ground_truth)

        # Calculate causal effect (change from baseline)
        causal_effect = 0.0
        if baseline_correct and not patched_correct:
            causal_effect = -1.0  # Negative effect: broke correct reasoning
        elif not baseline_correct and patched_correct:
            causal_effect = 1.0   # Positive effect: fixed incorrect reasoning
        # Otherwise: no change (0.0)

        # Calculate activation changes
        if 'original' in activations_cache and 'intervened' in activations_cache:
            # ✅ FIX: Add .detach().cpu() for activation tensors
            original = activations_cache['original'].detach().cpu().numpy()  # CHANGED
            intervened = activations_cache['intervened'].detach().cpu().numpy()  # CHANGED
            delta = np.abs(intervened - original)
            activation_delta_mean = float(np.mean(delta))
            activation_delta_max = float(np.max(delta))
        else:
            activation_delta_mean = 0.0
            activation_delta_max = 0.0

        if self.debug:
            effect_str = "🔻 broke" if causal_effect < 0 else "🔺 fixed" if causal_effect > 0 else "➡️ no change"
            print(f"Patched: {patched_prediction} (Correct: {patched_correct}) | {effect_str}")
            print(f"Activation Δμ: {activation_delta_mean:.4f}, Δmax: {activation_delta_max:.4f}\n")

        return PatchingResult(
            layer_idx=layer_to_patch,
            model_name=self.model_name,
            sample_id=sample_id,
            question=question,
            dataset_name=dataset_name,
            ground_truth=ground_truth,
            source_run="baseline",
            target_run="intervened",
            patched_prediction=str(patched_prediction) if patched_prediction else "None",
            patched_correct=patched_correct,
            causal_effect=causal_effect,
            intervention_type=intervention_type,
            activation_delta_mean=activation_delta_mean,
            activation_delta_max=activation_delta_max
        )
    def get_baseline_performance(self, question: str, ground_truth: Any) -> Tuple[Optional[float], bool]:
        """Get baseline performance without any intervention"""
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_prediction = self.validator.extract_answer(generated_text)
        baseline_correct = self.validator.validate(baseline_prediction, ground_truth)

        return baseline_prediction, baseline_correct

    def format_prompt(self, question: str) -> str:
        """Format prompt for patching experiments"""
        return f"Solve: {question}\nFinal Answer:"



class EmergenceArchaeologist:
    """Main research framework orchestrating all experiments"""

    def __init__(self, output_dir: str = "./emergence_results", debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.encoding_vs_expression = {} 
        # Create subdirectories
        (self.output_dir / "probing").mkdir(exist_ok=True)
        (self.output_dir / "patching").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        self.current_model_manager = None
        print(f"📁 Output directory: {self.output_dir}")



    def _get_or_load_model(self, model_name: str) -> SharedModelManager:
        """
        ✅ KEY FIX: Load model only if not already loaded
        """
        if self.current_model_manager is None or \
           self.current_model_manager.model_name != model_name:
            
            # Clean up old model if exists
            if self.current_model_manager is not None:
                print(f"🔄 Switching models: {self.current_model_manager.model_name} → {model_name}")
                self.current_model_manager.cleanup()
            
            # Load new model
            self.current_model_manager = SharedModelManager(
                model_name=model_name,
                debug=self.debug
            )
        else:
            print(f"♻️ Reusing already-loaded model: {model_name}")
        
        return self.current_model_manager


    def create_attention_visualizations(self, model_name: str):
        """
        Create publication-quality attention pattern visualizations.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load attention analysis
        attention_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_analysis.json"

        if not attention_path.exists():
            print(f"❌ No attention analysis found at {attention_path}")
            return

        with open(attention_path, 'r') as f:
            attention_data = json.load(f)

        layer_stats = attention_data.get('layer_stats', [])
        if not layer_stats:
            print("❌ No layer stats in attention analysis")
            return

        # ✅ FIX: Calculate threshold dynamically based on model size
        num_layers = len(layer_stats)
        if num_layers <= 16:  # Small model (1B-3B)
            threshold = 0.15  # 5%
        elif num_layers < 30:  # Medium model (5B-7B)
            threshold = 0.03  # 10%
        else:  # Large model (13B+)
            threshold = 0.05  # 15%
        
        print(f"📊 Visualization: Using threshold={threshold:.0%} for {num_layers}-layer model")

        # Extract data
        layers = [s['layer'] for s in layer_stats]
        math_attn = [s['avg_math_attention'] for s in layer_stats]
        reasoning_attn = [s['avg_reasoning_attention'] for s in layer_stats]
        specialized_heads = [s['num_consistent_specialized_heads'] for s in layer_stats]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Attention scores over layers
        ax = axes[0]
        ax.plot(layers, math_attn, 'o-', label='Math Attention', linewidth=2, markersize=6)
        ax.plot(layers, reasoning_attn, 's-', label='Reasoning Attention', linewidth=2, markersize=6)
        
        # ✅ FIX: Use dynamic threshold instead of hardcoded 0.15
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'Threshold ({threshold:.0%})')

        # Mark emergence layer
        emergence_layer = attention_data.get('emergence_layer', -1)
        if emergence_layer >= 0:
            ax.axvline(x=emergence_layer, color='g', linestyle='--', alpha=0.7,
                      label=f'Emergence Layer {emergence_layer}')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.set_title(f'Attention Pattern Evolution - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Specialized heads per layer
        ax = axes[1]
        ax.bar(layers, specialized_heads, alpha=0.7, color='steelblue')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Number of Specialized Heads', fontsize=12)
        ax.set_title('Math-Specialized Attention Heads by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_patterns.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Attention visualization saved to: {fig_path}")

        plt.close()


    def _analyze_attention_patterns(self, attention_results: List[List[AttentionAnalysisResult]],
                                   model_name: str) -> Dict:
        """
        Analyze aggregated attention patterns across all samples.

        Returns summary statistics.
        """
        if not attention_results:
            return {}

        # Use the analyzer's aggregation method
        analyzer = EnhancedAttentionAnalyzer(None, None)  # Dummy instance for aggregation
        aggregated = analyzer.aggregate_results(attention_results)

        # Add model-specific info
        aggregated['model_name'] = model_name
        aggregated['num_samples_analyzed'] = len(attention_results)

        return aggregated


    def load_dataset(self, dataset_name: str, split: str = None, num_samples: int = 50) -> List[Dict]:
        """Load various datasets for capability testing"""
        print(f"\n📚 Loading {dataset_name} dataset...")

        samples = []

        if dataset_name == "gsm8k":
            # GSM8K only has 'train' and 'test' splits
            if split is None:
                split = "test"  # Default to test split
            dataset = load_dataset("gsm8k", "main", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                question = item['question']
                answer_text = item['answer']
                match = re.search(r'####\s*(\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
                if match:
                    ground_truth = float(match.group(1).replace(',', ''))
                    samples.append({
                        'id': idx, 'question': question, 'ground_truth': ground_truth,
                        'dataset': dataset_name, 'task_type': 'math'
                    })

        elif dataset_name == "boolq":
            # BoolQ has 'train', 'validation', 'test' splits
            if split is None:
                split = "validation"  # Default to validation split
            dataset = load_dataset("boolq", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                samples.append({
                    'id': idx,
                    'question': f"Passage: {item['passage']}\nQuestion: {item['question']}",
                    'ground_truth': item['answer'],  # True/False
                    'dataset': dataset_name,
                    'task_type': 'reasoning'
                })

        elif dataset_name == "commonsense_qa":
            # Commonsense QA has 'train', 'validation', 'test' splits
            if split is None:
                split = "validation"  # Default to validation split
            dataset = load_dataset("commonsense_qa", split=split)
            for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                question = f"{item['question']}\nOptions: " + " ".join([f"({chr(65+i)}) {opt}"
                                                                      for i, opt in enumerate(item['choices']['text'])])
                # Convert answer key to index
                answer_key = item['answerKey']
                if answer_key in ['A', 'B', 'C', 'D', 'E']:
                    answer_idx = ord(answer_key) - 65
                    ground_truth = item['choices']['text'][answer_idx]
                else:
                    continue

                samples.append({
                    'id': idx, 'question': question, 'ground_truth': ground_truth,
                    'dataset': dataset_name, 'task_type': 'commonsense'
                })

        elif dataset_name == "hellaswag":
            if split is None:
                split = "validation"
            
            try:
                dataset = load_dataset("Rowan/hellaswag", split=split)
                
                for idx, item in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                    context = item['ctx']
                    endings = item['endings']
                    correct_idx = int(item['label'])
                    
                    # ✅ FIX: Store BOTH the number AND the text
                    # Use the LETTER (A, B, C, D) as ground truth for matching
                    ground_truth_letter = chr(65 + correct_idx)  # 0→A, 1→B, 2→C, 3→D
                    ground_truth_text = endings[correct_idx]
                    
                    # Format question with lettered options
                    question = f"Complete the following scenario:\n\n{context}\n\nWhich continuation makes the most sense?\n"
                    question += "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
                    
                    samples.append({
                        'id': idx,
                        'question': question,
                        'ground_truth': ground_truth_letter,  # ✅ Use letter: "A", "B", "C", or "D"
                        'ground_truth_text': ground_truth_text,  # Store full text for reference
                        'dataset': dataset_name,
                        'task_type': 'commonsense'
                    })
                    
            except Exception as e:
                print(f"❌ Error loading HellaSwag: {e}")
                return []


        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"✅ Loaded {len(samples)} samples from {dataset_name} ({split} split)")
        return samples


    def analyze_enhanced_detection(self, df: pd.DataFrame, model_name: str):
        """Compare single-token vs multi-token detection patterns"""
        print(f"\n🎯 ENHANCED DETECTION ANALYSIS")
        print("="*80)

        # Compare single-token vs multi-token by layer
        comparison_data = []
        for layer_idx in sorted(df['layer_idx'].unique()):
            layer_data = df[df['layer_idx'] == layer_idx]

            single_acc = layer_data['is_correct'].mean()
            multi_acc = layer_data['multi_token_correct'].mean()
            first_digit_acc = layer_data['first_digit_match'].mean()

            comparison_data.append({
              'layer': int(layer_idx),  # Convert to native int
              'single_token_accuracy': float(single_acc),  # Convert to float
              'multi_token_accuracy': float(multi_acc),
              'first_digit_accuracy': float(first_digit_acc),
              'improvement': float(multi_acc - single_acc)
          })

            # Print layer analysis
            improvement_str = f"(+{multi_acc-single_acc:.1%})" if multi_acc > single_acc else ""
            print(f"Layer {layer_idx:2d}: Single={single_acc:.1%} → Multi={multi_acc:.1%} {improvement_str}")

            # Flag major improvements
            if multi_acc > single_acc + 0.2:  # 20% improvement
                print(f"    🚨 MAJOR IMPROVEMENT: Multi-token found what single-token missed!")

        # Find emergence layers
        single_emergence = next((l for l in comparison_data if l['single_token_accuracy'] > 0.5), None)
        multi_emergence = next((l for l in comparison_data if l['multi_token_accuracy'] > 0.5), None)

        print(f"\n📊 EMERGENCE COMPARISON:")
        if single_emergence:
            print(f"  Single-token emergence: Layer {single_emergence['layer']} ({single_emergence['single_token_accuracy']:.1%})")
        if multi_emergence:
            print(f"  Multi-token emergence:  Layer {multi_emergence['layer']} ({multi_emergence['multi_token_accuracy']:.1%})")

        if multi_emergence and single_emergence and multi_emergence['layer'] < single_emergence['layer']:
            gap = single_emergence['layer'] - multi_emergence['layer']
            print(f"  ✅ Multi-token detects emergence {gap} layers EARLIER!")

        # Save enhanced analysis
        enhanced_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_enhanced_detection.json"
        with open(enhanced_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_compatible_data = json.loads(json.dumps(comparison_data, default=str))
            json.dump(json_compatible_data, f, indent=2)
        print(f"💾 Enhanced detection analysis saved to: {enhanced_path}")

        return comparison_data


    def run_probing_experiment(self, model_name: str, samples: List[Dict],
                            task_type: str = "math") -> pd.DataFrame:
        """Run layer-wise probing on all samples"""
        print(f"\n{'='*80}")
        print(f"🔬 PROBING EXPERIMENT: {model_name}")
        print(f"{'='*80}\n")
        manager = self._get_or_load_model(model_name)
        prober = LayerWiseProber(manager, debug=self.debug)

        all_results = []
        successful_samples = 0

        for sample in tqdm(samples, desc="Probing samples"):
            try:
                layer_results = prober.probe_all_layers(
                    question=sample['question'],
                    ground_truth=sample['ground_truth'],
                    sample_id=sample['id'],
                    task_type=task_type,
                    dataset_name=sample['dataset']
                )

                # DEBUG: Check if we got any results
                if layer_results:
                    all_results.extend(layer_results)
                    successful_samples += 1
                    if self.debug:
                        print(f"✅ Sample {sample['id']}: Got {len(layer_results)} layer results")
                else:
                    print(f"⚠️ Sample {sample['id']}: No layer results returned")

            except Exception as e:
                print(f"❌ Error on sample {sample['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"📊 Probing completed: {successful_samples}/{len(samples)} samples successful")

        if not all_results:
            print("❌ No probing results collected!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])
        prober.probing_results_df = df

        # DEBUG: Print DataFrame info
        print(f"📋 Results DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📊 Columns: {list(df.columns)}")

        # ✅ SIMPLE ENCODING vs EXPRESSION ANALYSIS
        print(f"\n🔍 SIMPLE ENCODING vs EXPRESSION ANALYSIS:")
        try:
            encoding_vs_expression = prober.linear_prober.compare_encoding_vs_expression_v2(df)

            # ✅ CRITICAL: Store for use in analyze_emergence_patterns
            self.encoding_vs_expression = encoding_vs_expression

            early_encoding_detected = False
            for layer_idx, results in encoding_vs_expression.items():
                expr_acc = results['expression_accuracy']
                enc_acc = results['encoding_accuracy']
                gap = results['gap']

                if enc_acc is not None:
                    print(f"   Layer {layer_idx:2d}: Expression={expr_acc:.1%}, Encoding={enc_acc:.1%}, Gap={gap:+.1%}")

                    # Check for early encoding detection
                    if layer_idx < prober.num_layers // 2 and enc_acc > expr_acc + 0.3:
                        early_encoding_detected = True
                        print(f"   🎯 EARLY ENCODING DETECTED at layer {layer_idx}!")

            if early_encoding_detected:
                print(f"\n🚨 CRITICAL FINDING: Capabilities are ENCODED in middle layers but not EXPRESSED!")

            # Save encoding analysis
            encoding_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_encoding_analysis.json"
            with open(encoding_path, 'w') as f:
                json.dump(encoding_vs_expression, f, indent=2)
            print(f"💾 Encoding vs Expression analysis saved to: {encoding_path}")

        except Exception as e:
            print(f"⚠️ Encoding vs Expression analysis failed: {e}")

        # ✅ ADD MULTI-TOKEN ANALYSIS
        print(f"\n🎯 ANALYZING MULTI-TOKEN PROBING RESULTS...")
        multi_token_analysis = self._analyze_multi_token_results(df)

        # ✅ NEW: ATTENTION ANALYSIS
        print(f"\n🔍 ANALYZING ATTENTION PATTERNS...")
        try:
            if hasattr(prober, '_attention_results') and prober._attention_results:
                attention_analysis = self._analyze_attention_patterns(
                    prober._attention_results, model_name
                )

                # Save attention analysis
                attention_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_attention_analysis.json"
                with open(attention_path, 'w') as f:
                    json.dump(attention_analysis, f, indent=2)
                print(f"💾 Attention analysis saved to: {attention_path}")

                # Print key findings
                if attention_analysis:
                    print(f"\n📊 ATTENTION FINDINGS:")
                    print(f"   Attention emergence layer: {attention_analysis.get('emergence_layer', 'N/A')}")

                    # Find layer with most specialized heads
                    layer_stats = attention_analysis.get('layer_stats', [])
                    if layer_stats:
                        max_heads_layer = max(layer_stats,
                                            key=lambda x: x['num_consistent_specialized_heads'])
                        print(f"   Peak specialized heads: Layer {max_heads_layer['layer']} "
                              f"({max_heads_layer['num_consistent_specialized_heads']} heads)")
                        print(f"   Math attention at peak: {max_heads_layer['avg_math_attention']:.1%}")
            else:
                print("   ⚠️ No attention results available")

        except Exception as e:
            print(f"   ❌ Attention analysis failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

        if multi_token_analysis:
            # print(f"📊 MULTI-TOKEN FINDINGS:")
            # print(f"   Samples where multi-token found capabilities earlier: {multi_token_analysis['early_detection_samples']}")
            # print(f"   Layers with multi-token advantage: {multi_token_analysis['advantage_layers']}")
            # print(f"   First digit match success rate: {multi_token_analysis['first_digit_success']:.1%}")

            # Save multi-token analysis
            multi_token_path = self.output_dir / "analysis" / f"{model_name.replace('/', '_')}_multi_token_analysis.json"
            with open(multi_token_path, 'w') as f:
                json.dump(multi_token_analysis, f, indent=2)
            print(f"💾 Multi-token analysis saved to: {multi_token_path}")


        # # ✅ ADD ENHANCED DETECTION ANALYSIS
        # print(f"\n🔬 COMPREHENSIVE DETECTION COMPARISON...")
        # enhanced_comparison = self.analyze_enhanced_detection(df, model_name)

        # # Check for critical findings
        # early_multi_detection = any(
        #     layer['multi_token_accuracy'] > layer['single_token_accuracy'] + 0.3
        #     and layer['layer'] < prober.num_layers // 2
        #     for layer in enhanced_comparison
        # )

        # if early_multi_detection:
        #     print(f"\n🚨 CRITICAL FINDING: Multi-token probing reveals capabilities in EARLY layers!")
        #     print(f"   Your single-token method was missing early-stage capability formation!")

        # Save results
        model_safe = model_name.replace('/', '_')
        output_path = self.output_dir / "probing" / f"{model_safe}_probing.csv"
        df.to_csv(output_path, index=False)
        print(f"\n💾 Probing results saved to: {output_path}")

        return df

    def _analyze_multi_token_results(self, df: pd.DataFrame) -> dict:
        """Analyze multi-token probing results vs single-token"""
        if 'multi_token_correct' not in df.columns:
            return {}

        analysis = {
            'early_detection_samples': 0,
            'advantage_layers': [],
            'first_digit_success': 0.0,
            'layer_breakdown': {}
        }

        # Analyze by layer
        for layer_idx in sorted(df['layer_idx'].unique()):
            layer_data = df[df['layer_idx'] == layer_idx]

            single_token_acc = layer_data['is_correct'].mean()
            multi_token_acc = layer_data['multi_token_correct'].mean()
            first_digit_acc = layer_data['first_digit_match'].mean()

            analysis['layer_breakdown'][int(layer_idx)] = {  # Convert key to native int
                'single_token_accuracy': float(single_token_acc),  # Convert to float
                'multi_token_accuracy': float(multi_token_acc),
                'first_digit_accuracy': float(first_digit_acc),
                'multi_token_advantage': float(multi_token_acc - single_token_acc)
            }

            # Count layers where multi-token has advantage
            # Count layers where multi-token has advantage
            if multi_token_acc > single_token_acc + 0.1:  # 10% advantage
                analysis['advantage_layers'].append(int(layer_idx))  # Convert to native int

        # Count samples where multi-token detected capabilities earlier
        sample_analysis = df.groupby('sample_id').apply(
            lambda x: self._find_earliest_detection(x)
        )
        analysis['early_detection_samples'] = int(sample_analysis.sum())  # Convert to native int
        # First digit success rate
        analysis['first_digit_success'] = df['first_digit_match'].mean()

        return analysis

    def _find_earliest_detection(self, sample_data: pd.DataFrame) -> bool:
        """Check if multi-token detected capability earlier than single-token for a sample"""
        single_token_layers = sample_data[sample_data['is_correct'] == True]['layer_idx']
        multi_token_layers = sample_data[sample_data['multi_token_correct'] == True]['layer_idx']

        if len(multi_token_layers) == 0:
            return False

        if len(single_token_layers) == 0:
            return True  # Multi-token found it when single-token didn't

        earliest_single = min(single_token_layers)
        earliest_multi = min(multi_token_layers)

        return earliest_multi < earliest_single

    def run_patching_experiment(self, model_name: str, samples: List[Dict],
                                layers_to_test: Optional[List[int]] = None,
                                intervention_type: str = "zero_ablate") -> pd.DataFrame:
        """Run activation patching experiments"""
        print(f"\n{'='*80}")
        print(f"🔧 PATCHING EXPERIMENT: {model_name}")
        print(f"{'='*80}\n")


        manager = self._get_or_load_model(model_name)

        patcher = ActivationPatcher(manager, debug=self.debug)

        if layers_to_test is None:
            # Test every 4th layer for efficiency
            layers_to_test = list(range(0, patcher.num_layers, 4))

        print(f"Testing {len(layers_to_test)} layers: {layers_to_test}")

        all_results = []
        for sample in tqdm(samples[:10], desc="Patching samples"):  # Limit samples for patching
            for layer_idx in layers_to_test:
                try:
                    result = patcher.patch_layer_activation(
                        question=sample['question'],
                        ground_truth=sample['ground_truth'],
                        sample_id=sample['id'],
                        layer_to_patch=layer_idx,
                        intervention_type=intervention_type,
                        dataset_name=sample['dataset']
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error patching layer {layer_idx} on sample {sample['id']}: {e}")
                    continue

        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])

        # Save results
        model_safe = model_name.replace('/', '_')
        output_path = self.output_dir / "patching" / f"{model_safe}_patching_{intervention_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n💾 Patching results saved to: {output_path}")

        return df

    def analyze_emergence_patterns(self, probing_df: pd.DataFrame) -> Dict:
        """Analyze where and how capabilities emerge across layers"""
        print(f"\n{'='*80}")
        print(f"📊 EMERGENCE ANALYSIS")
        print(f"{'='*80}\n")
    
        print(f"📋 DataFrame columns: {list(probing_df.columns)}")
        print(f"📊 DataFrame shape: {probing_df.shape}")
    
        if probing_df.empty:
            print("❌ Empty DataFrame - skipping analysis")
            return {}
    
        required_columns = ['layer_idx', 'is_correct', 'layer_confidence']
        missing_columns = [col for col in required_columns if col not in probing_df.columns]
    
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"✅ Available columns: {list(probing_df.columns)}")
            return {}
    
        analysis = {}
    
        # 1. Layer-wise accuracy
        layer_stats = probing_df.groupby('layer_idx').agg({
            'is_correct': 'mean',
            'layer_confidence': 'mean',
            'activation_mean': 'mean',
            'activation_std': 'mean',
            'activation_sparsity': 'mean'
        }).reset_index()
    
        layer_stats.columns = ['layer', 'accuracy', 'confidence', 'act_mean', 'act_std', 'sparsity']
    
        print("Layer-wise Statistics:")
        print(layer_stats.round(4).to_string())
    
        # 2. Check for encoding vs expression results from linear probes
        encoding_emergence_layer = -1
        expression_emergence_layer = -1
        
        if hasattr(self, 'encoding_vs_expression') and self.encoding_vs_expression:
            print(f"\n🔬 DETECTING EMERGENCE FROM LINEAR PROBES:")
            
            for layer_idx in sorted(self.encoding_vs_expression.keys()):
                enc_acc = self.encoding_vs_expression[layer_idx].get('encoding_accuracy')
                expr_acc = self.encoding_vs_expression[layer_idx].get('expression_accuracy')
                
                # Emergence = first layer with >70% encoding accuracy
                if enc_acc is not None and enc_acc > 0.7 and encoding_emergence_layer == -1:
                    encoding_emergence_layer = layer_idx
                    print(f"   📍 ENCODING emergence at layer {layer_idx} ({enc_acc:.1%} linear probe accuracy)")
                
                # Expression emergence = first layer with >70% expression accuracy  
                if expr_acc is not None and expr_acc > 0.7 and expression_emergence_layer == -1:
                    expression_emergence_layer = layer_idx
                    print(f"   📍 EXPRESSION emergence at layer {layer_idx} ({expr_acc:.1%} next-token accuracy)")
            
            if encoding_emergence_layer > 0 and expression_emergence_layer > 0:
                gap = expression_emergence_layer - encoding_emergence_layer
                print(f"   🎯 GAP: {gap} layers between where capability is computed vs expressed")
            elif encoding_emergence_layer > 0:
                print(f"   ⚠️ Capability is ENCODED at layer {encoding_emergence_layer} but never strongly EXPRESSED")
    
        # 3. Crystallization layer detection (improved)
        print(f"\n🔍 DEBUG: Detailed layer accuracy progression:")
        for layer in range(len(layer_stats)):
            layer_data = probing_df[probing_df['layer_idx'] == layer]
            if len(layer_data) > 0:
                acc = layer_data['is_correct'].mean()
                conf = layer_data['layer_confidence'].mean()
                sample_correctness = [f"{sample_id}:{correct}" for sample_id, correct in
                                    zip(layer_data['sample_id'], layer_data['is_correct'])]
                print(f"Layer {layer:2d}: accuracy={acc:.3f}, confidence={conf:.3f}, samples={sample_correctness}")
    
        crystallization_layer = -1
        detection_method = "not_found"
    
        # More lenient thresholds
        sustained_threshold = 0.7  # 70% accuracy (was 80%)
        min_sustained_layers = 3
        confidence_requirement = 0.01
    
        num_layers = len(layer_stats)
        skip_threshold = max(3, num_layers // 10)  # Skip only first 10%
    
        print(f"🔍 Crystallization search: Skipping layers 0-{skip_threshold-1}")
    
        # Strategy 1: Sustained high accuracy
        for i in range(skip_threshold, len(layer_stats) - min_sustained_layers + 1):
            window = layer_stats.iloc[i:i + min_sustained_layers]
            window_confidences = layer_stats['confidence'].iloc[i:i + min_sustained_layers]
    
            if (all(window['accuracy'] >= sustained_threshold) and
                all(window_confidences >= confidence_requirement)):
                crystallization_layer = i
                detection_method = 'sustained_high_accuracy'
                print(f"✅ Found crystallization at layer {i} via sustained accuracy")
                break
    
        # Strategy 2: Significant jump
        if crystallization_layer == -1:
            accuracy_jump_threshold = 0.3
            for i in range(skip_threshold, len(layer_stats)):
                if i == 0:
                    continue
                prev_acc = layer_stats['accuracy'].iloc[i-1]
                curr_acc = layer_stats['accuracy'].iloc[i]
                curr_conf = layer_stats['confidence'].iloc[i]
    
                if (curr_acc - prev_acc >= accuracy_jump_threshold and
                    curr_conf >= 0.01 and curr_acc >= 0.5):
                    crystallization_layer = i
                    detection_method = 'significant_jump'
                    print(f"✅ Found crystallization at layer {i} via significant jump")
                    break
    
        # Strategy 3: Fallback
        if crystallization_layer == -1:
            late_layers_stats = layer_stats.iloc[skip_threshold:]
            if len(late_layers_stats) > 0:
                max_acc_idx = late_layers_stats['accuracy'].idxmax()
                crystallization_layer = max_acc_idx
                detection_method = 'max_accuracy_fallback'
                print(f"✅ Found crystallization at layer {crystallization_layer} via fallback")
    
        # Validate crystallization
        if crystallization_layer >= 0:
            crystal_acc = layer_stats['accuracy'].iloc[crystallization_layer]
            crystal_conf = layer_stats['confidence'].iloc[crystallization_layer]
            print(f"✅ Validated crystallization: Layer {crystallization_layer}, "
                  f"Accuracy={crystal_acc:.3f}, Confidence={crystal_conf:.3f}")
    
        # 4. Calculate capability evolution
        early_layers = probing_df[probing_df['layer_idx'] < num_layers // 4]
        mid_layers = probing_df[(probing_df['layer_idx'] >= num_layers // 4) &
                               (probing_df['layer_idx'] < num_layers // 2)]
        late_layers = probing_df[probing_df['layer_idx'] >= num_layers // 2]
    
        print(f"\n📊 Layer grouping: Early (0-{num_layers//4-1}), "
              f"Mid ({num_layers//4}-{num_layers//2-1}), Late ({num_layers//2}-{num_layers-1})")
    
        # Store results
        analysis['layer_statistics'] = layer_stats.to_dict('records')
        analysis['crystallization_layer'] = int(crystallization_layer)
        analysis['crystallization_method'] = detection_method
        analysis['encoding_emergence_layer'] = int(encoding_emergence_layer) if encoding_emergence_layer > 0 else None
        analysis['expression_emergence_layer'] = int(expression_emergence_layer) if expression_emergence_layer > 0 else None
        analysis['num_layers'] = int(probing_df['layer_idx'].max() + 1)
        analysis['overall_accuracy'] = float(layer_stats['accuracy'].iloc[-1])
    
        analysis['capability_evolution'] = {
            'early_accuracy': float(early_layers['is_correct'].mean()) if len(early_layers) > 0 else 0.0,
            'mid_accuracy': float(mid_layers['is_correct'].mean()) if len(mid_layers) > 0 else 0.0,
            'late_accuracy': float(late_layers['is_correct'].mean()) if len(late_layers) > 0 else 0.0,
            'early_confidence': float(early_layers['layer_confidence'].mean()) if len(early_layers) > 0 else 0.0,
            'late_confidence': float(late_layers['layer_confidence'].mean()) if len(late_layers) > 0 else 0.0,
        }
    
        analysis['activation_evolution'] = {
            'early_sparsity': float(early_layers['activation_sparsity'].mean()) if len(early_layers) > 0 else 0.0,
            'late_sparsity': float(late_layers['activation_sparsity'].mean()) if len(late_layers) > 0 else 0.0,
            'activation_growth': float(late_layers['activation_std'].mean()) /
                               float(early_layers['activation_std'].mean()) if len(early_layers) > 0 else 1.0
        }
    
        # Warnings
        print(f"\n🎯 KEY FINDINGS:")
        
        if crystallization_layer >= num_layers - 2:
            print(f"  ⚠️ WARNING: Crystallization at layer {crystallization_layer}/{num_layers} (final layer!)")
            print(f"     This suggests detection is only catching the OUTPUT, not computation")
            if encoding_emergence_layer > 0:
                print(f"     TRUE emergence is likely at layer {encoding_emergence_layer} (from linear probes)")
        
        if encoding_emergence_layer > 0:
            print(f"  - Encoding emergence: Layer {encoding_emergence_layer} ({encoding_emergence_layer/num_layers*100:.0f}% depth)")
        if expression_emergence_layer > 0:
            print(f"  - Expression emergence: Layer {expression_emergence_layer} ({expression_emergence_layer/num_layers*100:.0f}% depth)")
        print(f"  - Crystallization layer: {crystallization_layer} (method: {detection_method})")
        print(f"  - Accuracy progression: {analysis['capability_evolution']['early_accuracy']:.1%} → "
              f"{analysis['capability_evolution']['mid_accuracy']:.1%} → "
              f"{analysis['capability_evolution']['late_accuracy']:.1%}")
        print(f"  - Confidence growth: {analysis['capability_evolution']['early_confidence']:.3f} → "
              f"{analysis['capability_evolution']['late_confidence']:.3f}")
        print(f"  - Final accuracy: {analysis['overall_accuracy']:.2%}")
        print(f"  - Activation sparsity: {analysis['activation_evolution']['early_sparsity']:.1%} → "
              f"{analysis['activation_evolution']['late_sparsity']:.1%}")
    
        # Save analysis
        analysis_path = self.output_dir / "analysis" / "emergence_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Analysis saved to: {analysis_path}")
    
        return analysis

    def cleanup_models(self):
        """
        ✅ NEW: Clean up models when completely done
        """
        if self.current_model_manager is not None:
            self.current_model_manager.cleanup()
            self.current_model_manager = None


    def create_periodic_table(self, multi_model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a 'periodic table' of capability emergence across models and tasks
        """
        print(f"\n{'='*80}")
        print(f"📋 CREATING PERIODIC TABLE OF EMERGENCE")
        print(f"{'='*80}\n")

        periodic_data = []

        for model_name, probing_df in multi_model_results.items():
            print(f"Analyzing {model_name}...")

            # Skip if DataFrame is empty
            if probing_df.empty:
                print(f"❌ Skipping {model_name} - empty DataFrame")
                continue

            analysis = self.analyze_emergence_patterns(probing_df)

            # Skip if analysis failed
            if not analysis:
                print(f"❌ Skipping {model_name} - analysis failed")
                continue

            periodic_data.append({
                'model': model_name,
                'num_layers': analysis['num_layers'],
                'crystallization_layer': analysis['crystallization_layer'],
                'crystallization_depth_pct': (analysis['crystallization_layer'] / analysis['num_layers'] * 100)
                                              if analysis['crystallization_layer'] > 0 else 0,
                'final_accuracy': analysis['overall_accuracy'],
                'early_sparsity': analysis['activation_evolution']['early_sparsity'],
                'late_sparsity': analysis['activation_evolution']['late_sparsity']
            })

        if not periodic_data:
            print("❌ No valid data for periodic table")
            return pd.DataFrame()

        periodic_table = pd.DataFrame(periodic_data)

        print("\n📊 PERIODIC TABLE OF CAPABILITY EMERGENCE:")
        print("="*80)
        print(periodic_table.to_string(index=False))
        print("="*80)

        # Save
        table_path = self.output_dir / "analysis" / "periodic_table.csv"
        periodic_table.to_csv(table_path, index=False)
        print(f"\n💾 Periodic table saved to: {table_path}")

        return periodic_table






def main():
    """Main research pipeline - FIXED TO ACCUMULATE ALL DATASETS"""
    from huggingface_hub import login
    login(token="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")
    
    MODELS_TO_TEST = [
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct", 
        # "prithivMLmods/Llama-3.1-5B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "google/gemma-2-2b-it"
    ]

    DATASETS_TO_TEST = [
        ("gsm8k", "math", "test"),
        ("boolq", "reasoning", "validation"),
        ("commonsense_qa", "commonsense", "validation"),
        ("hellaswag", "commonsense", "validation"),      
    ]
    
    NUM_SAMPLES = 400
    DEBUG = False
    
    # Initialize researcher
    archaeologist = EmergenceArchaeologist(
        output_dir="./emergence_archaeology_results",
        debug=DEBUG
    )
    
    all_results = {}
    
    # ✅ FIX: Loop over MODELS first, then ACCUMULATE all datasets for each model
    for model_name in MODELS_TO_TEST:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        # ✅ ACCUMULATE ALL DATASETS FOR THIS MODEL
        all_probing_dfs = []
        
        for dataset_name, task_type, split in DATASETS_TO_TEST:
            print(f"\n--- Testing {model_name} on {dataset_name} ---")
            
            samples = archaeologist.load_dataset(
                dataset_name=dataset_name,
                split=split,
                num_samples=NUM_SAMPLES
            )
            
            try:
                # Run probing for THIS dataset
                probing_df = archaeologist.run_probing_experiment(
                    model_name=model_name,
                    samples=samples,
                    task_type=task_type
                )
                
                # ✅ ADD dataset identifier column
                probing_df['dataset_name'] = dataset_name
                probing_df['task_type'] = task_type
                
                # ✅ ACCUMULATE (don't overwrite!)
                all_probing_dfs.append(probing_df)
                
                print(f"✅ Completed {dataset_name}: {len(probing_df)} rows")
                
            except Exception as e:
                print(f"❌ Error on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ✅ COMBINE all datasets for this model
        if all_probing_dfs:
            combined_df = pd.concat(all_probing_dfs, ignore_index=True)
            
            print(f"\n📊 COMBINED RESULTS FOR {model_name}:")
            print(f"   Total samples: {combined_df['sample_id'].nunique()}")
            print(f"   Total rows: {len(combined_df)}")
            print(f"   Datasets: {combined_df['dataset_name'].unique()}")
            
            # ✅ SAVE combined data (overwrites per-dataset files)
            model_safe = model_name.replace('/', '_')
            output_path = archaeologist.output_dir / "probing" / f"{model_safe}_probing_combined.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"💾 Saved combined data to: {output_path}")
            
            # ✅ ANALYZE combined data
            analysis = archaeologist.analyze_emergence_patterns(combined_df)
            
            # ✅ Store combined results
            all_results[f"{model_name}_combined"] = combined_df
            
            # ✅ OPTIONAL: Per-dataset analysis
            print(f"\n📊 Per-Dataset Emergence:")
            for dataset_name in combined_df['dataset_name'].unique():
                dataset_df = combined_df[combined_df['dataset_name'] == dataset_name]
                dataset_analysis = archaeologist.analyze_emergence_patterns(dataset_df)
                
                if dataset_analysis:
                    print(f"   {dataset_name}: Layer {dataset_analysis['crystallization_layer']} "
                          f"({dataset_analysis['crystallization_layer']/dataset_analysis['num_layers']*100:.1f}%)")
                    
                    # Save per-dataset analysis
                    dataset_analysis_path = archaeologist.output_dir / "analysis" / f"{model_safe}_{dataset_name}_emergence.json"
                    with open(dataset_analysis_path, 'w') as f:
                        json.dump(dataset_analysis, f, indent=2)
            
            # ✅ Run patching on COMBINED data (optional)
            if len(combined_df) >= 30:  # Need at least 30 samples
                try:
                    model_num_layers = len(archaeologist.current_model_manager.model.model.layers)
                    auto_patch_layers = calculate_patch_layers(
                        num_layers=model_num_layers,
                        num_patches=5
                    )
                    
                    # Use first 10 samples from combined data
                    patch_samples = []
                    for dataset_name in combined_df['dataset_name'].unique()[:2]:  # First 2 datasets
                        dataset_samples = combined_df[combined_df['dataset_name'] == dataset_name].head(5)
                        for _, row in dataset_samples.iterrows():
                            patch_samples.append({
                                'id': row['sample_id'],
                                'question': row['question'],
                                'ground_truth': row['ground_truth'],
                                'dataset': row['dataset_name']
                            })
                    
                    if patch_samples:
                        patching_df = archaeologist.run_patching_experiment(
                            model_name=model_name,
                            samples=patch_samples,
                            layers_to_test=auto_patch_layers,  
                            intervention_type="residual_patch"
                        )
                except Exception as e:
                    print(f"⚠️ Patching failed: {e}")
        
        # ✅ Cleanup model after processing all datasets
        # (Will be reloaded for next model)
    
    # ✅ Final cleanup when completely done
    archaeologist.cleanup_models()
    
    # Create final analysis
    if all_results:
        archaeologist.create_periodic_table(all_results)
    
    # ✅ Create attention visualizations
    print(f"\n📊 CREATING ATTENTION VISUALIZATIONS...")
    for model_name in MODELS_TO_TEST:
        try:
            archaeologist.create_attention_visualizations(model_name)
        except Exception as e:
            print(f"⚠️ Failed to create attention viz for {model_name}: {e}")

    print(f"\n\n{'#'*80}")
    print(f"# EMERGENCE ARCHAEOLOGY COMPLETE")
    print(f"# Results saved to: {archaeologist.output_dir}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()