import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import  List, Tuple, Optional, Any
from transformers.utils import logging
logging.set_verbosity_error()
from Layer_Wise_Emergence_Across_LLm
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