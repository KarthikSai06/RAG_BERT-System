import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

class Generator:
    def __init__(self):
        print(f"Loading LLM Generator: {config.LLM_MODEL_ID}...")
        self.model = None
        self.tokenizer = None
        if torch is None or AutoTokenizer is None:
            if config.ALLOW_LIGHTWEIGHT_FALLBACK:
                print("Transformers unavailable. Using extractive generator fallback.")
                return
            raise RuntimeError("Install torch and transformers to use the generator.")
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
            use_quant = config.USE_4BIT and torch.cuda.is_available() and config.USE_GPU
            if use_quant:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.LLM_MODEL_ID,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                print("Warning: No GPU / quantization disabled. High RAM required.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.LLM_MODEL_ID,
                    device_map="auto" if self.device.type == "cuda" else None,
                    torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
                )
        except Exception as exc:
            if not config.ALLOW_LIGHTWEIGHT_FALLBACK:
                raise
            print(f"LLM unavailable ({exc}). Using extractive generator fallback.")
            self.model = None
            self.tokenizer = None
            
    def generate(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return "I don't have that information in my documents."
        if self.model is None:
            return context_chunks[0]["text"]
            
        context_str = "\n\n".join(
            [f"[Passage {i+1}] {c['text']}" for i, c in enumerate(context_chunks)]
        )
        
        # Use chat template for correct [INST] token wrapping (Mistral / LLaMA-3)
        messages = [
            {
                "role": "user",
                "content": (
                    "You are an institutional helpdesk assistant. "
                    "Answer the question using ONLY the provided context. "
                    "If the context does not contain enough information, say "
                    "\"I don't have that information in my documents.\" "
                    "Do NOT use outside knowledge.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {query}"
                )
            }
        ]
        
        # apply_chat_template adds proper <s>[INST] / <|im_start|> tokens
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
