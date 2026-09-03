import pandas as pd
import torch
import config
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

def train():
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    
    train_df["label"] = train_df["label"].map(config.LABEL2ID)
    test_df["label"] = test_df["label"].map(config.LABEL2ID)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    tokenizer = BertTokenizer.from_pretrained(config.BERT_BASE_MODEL)
    
    def tokenize(batch):
        return tokenizer(batch["query"], padding="max_length", truncation=True, max_length=config.BERT_MAX_LEN)
        
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    model = BertForSequenceClassification.from_pretrained(
        config.BERT_BASE_MODEL,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH_SIZE,
        per_device_eval_batch_size=config.BERT_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.BERT_LR,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Training BERT...")
    trainer.train()
    
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")
    
    print(f"Saving model to {config.BERT_MODEL_DIR}")
    model.save_pretrained(config.BERT_MODEL_DIR)
    tokenizer.save_pretrained(config.BERT_MODEL_DIR)

if __name__ == "__main__":
    train()
