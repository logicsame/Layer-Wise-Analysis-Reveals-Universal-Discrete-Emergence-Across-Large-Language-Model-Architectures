import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from transformers.utils import logging
logging.set_verbosity_error()

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