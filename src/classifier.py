import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
except ImportError:
    torch = None
    BertTokenizer = None
    BertForSequenceClassification = None

class QueryClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        if torch is None or BertTokenizer is None:
            if config.ALLOW_LIGHTWEIGHT_FALLBACK:
                print("Transformers unavailable. Using the lightweight domain classifier.")
                return
            raise RuntimeError("Install torch and transformers to use the BERT classifier.")
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        print(f"Loading BERT Classifier on {self.device}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_DIR)
            self.model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL_DIR)
        except Exception:
            try:
                print("Fine-tuned classifier unavailable. Loading the base BERT classifier.")
                self.tokenizer = BertTokenizer.from_pretrained(config.BERT_BASE_MODEL)
                self.model = BertForSequenceClassification.from_pretrained(
                    config.BERT_BASE_MODEL,
                    num_labels=config.NUM_LABELS,
                    id2label=config.ID2LABEL,
                    label2id=config.LABEL2ID
                )
            except Exception:
                if not config.ALLOW_LIGHTWEIGHT_FALLBACK:
                    raise
                print("BERT weights unavailable. Using the lightweight domain classifier.")
                self.model = None
                self.tokenizer = None
        if self.model is None:
            return
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _fallback_predict(query):
        text = query.lower().strip()
        if not text:
            return "out-of-domain"
        domain_terms = (
            "admission", "fee", "fees", "payment", "scholarship", "hostel",
            "exam", "examination", "attendance", "semester", "waiver",
            "curfew", "refund", "portal", "student", "college", "university",
        )
        if not any(term in text for term in domain_terms):
            return "out-of-domain"
        procedural_terms = ("how", "apply", "procedure", "process", "where", "submit", "register", "pay")
        return "in-domain-procedural" if any(term in text for term in procedural_terms) else "in-domain-factual"

    def predict(self, query):
        if self.model is None:
            return self._fallback_predict(query)
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True,
            truncation=True, max_length=config.BERT_MAX_LEN
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
            
        return config.ID2LABEL[pred_id]

    def predict_with_confidence(self, query):
        """Returns (label, confidence_score) tuple."""
        if self.model is None:
            label = self._fallback_predict(query)
            return label, (0.95 if label != "out-of-domain" else 0.9)
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True,
            truncation=True, max_length=config.BERT_MAX_LEN
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
            
        return config.ID2LABEL[pred_id], round(confidence, 4)
