import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

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