import torch
from typing import Any, Tuple, Optional
from Layer_Wise_Emergence_Across_LLm.core.model_manager import SharedModelManager
from Layer_Wise_Emergence_Across_LLm.core.validation.validator import MultiTaskValidator, MathValidator
from Layer_Wise_Emergence_Across_LLm.core.validation.validator import PatchingResult
import numpy as np

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