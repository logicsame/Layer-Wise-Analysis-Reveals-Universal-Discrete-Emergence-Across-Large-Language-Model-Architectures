import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import pandas as pd

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