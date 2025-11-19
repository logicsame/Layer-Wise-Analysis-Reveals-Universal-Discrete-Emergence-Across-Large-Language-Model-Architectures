import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import re


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