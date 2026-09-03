import torch
import config
from transformers import BertTokenizer, BertForSequenceClassification

class QueryClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        print(f"Loading BERT Classifier on {self.device}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_DIR)
            self.model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL_DIR)
        except Exception as e:
            print("Model not found in BERT_MODEL_DIR. Using base model (for testing only).")
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_BASE_MODEL)
            self.model = BertForSequenceClassification.from_pretrained(
                config.BERT_BASE_MODEL, 
                num_labels=config.NUM_LABELS,
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID
            )
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=config.BERT_MAX_LEN)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=1).item()
            
        return config.ID2LABEL[pred_id]
