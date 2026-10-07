# TruthForge.py  -  Multilingual Fake News Detection with Political Context + XAI


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# AI libraries
try:
    from captum.attr import LayerIntegratedGradients
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("Warning: captum is not installed. To enable XAI run: pip install captum")



# 1. CONFIG

class Config:
    model_name = "xlm-roberta-base"      # For faster testing use: "distilbert-base-multilingual-cased"
    max_length = 128
    batch_size = 4
    epochs = 1
    lr = 2e-5
    weight_decay = 0.01
    metadata_dim = 32
    num_labels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    use_real_data = True
    real_data_path = "truthforge_data.json"

torch.manual_seed(Config.seed)
print(f"Using device: {Config.device}")



# 2. DATA LOADING

def create_synthetic_data(n_samples: int = 80):
    """Create synthetic data for testing"""
    languages = ["en", "bn", "hi"]
    parties = ["BJP", "Congress", "Democrat", "Republican", "None"]
    roles = ["politician", "journalist", "citizen", "unknown"]
    countries = ["IN", "BD", "US", "UNK"]

    texts, labels, metadata_list = [], [], []

    fake_templates = [
        "Breaking: {party} leader caught in major scandal involving corruption.",
        "Shocking revelation: Election results were manipulated by {party}.",
        "Urgent: Government of {country} hides truth about economic crisis.",
        "Fake news alert: {party} spreads misinformation about opposition."
    ]
    real_templates = [
        "Official statement from {party} regarding recent policy changes.",
        "Fact-checked report confirms the claims made by the government of {country}.",
        "Independent journalists verify the information shared by {role}.",
        "Transparent analysis of the economic data released by authorities."
    ]

    for i in range(n_samples):
        is_fake = i % 2 == 0
        party = np.random.choice(parties)
        role = np.random.choice(roles)
        country = np.random.choice(countries)
        lang = np.random.choice(languages)

        if is_fake:
            text = np.random.choice(fake_templates).format(party=party, country=country, role=role)
            label = 1
        else:
            text = np.random.choice(real_templates).format(party=party, country=country, role=role)
            label = 0

        text = f"[{lang}] {text}"
        texts.append(text)
        labels.append(label)
        metadata_list.append({
            "political_party": party,
            "speaker_role": role,
            "country": country,
            "source_credibility": round(np.random.uniform(0.1, 0.95), 2),
            "domain_age": int(np.random.randint(1, 20))
        })

    return texts, labels, metadata_list


def load_real_data(json_path: str = "truthforge_data.json"):
    """Load real data from JSON file"""
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []
    metadata_list = []

    for item in data:
        texts.append(str(item["text"]))
        labels.append(int(item["label"]))

        meta = item.get("metadata", {})
        metadata_list.append({
            "political_party": str(meta.get("political_party", "None")),
            "speaker_role": str(meta.get("speaker_role", "unknown")),
            "country": str(meta.get("country", "UNK")),
            "source_credibility": float(meta.get("source_credibility", 0.5)),
            "domain_age": int(meta.get("domain_age", 0))
        })

    print(f"Total samples loaded: {len(texts)}")
    print(f"Fake samples: {sum(labels)} | Real samples: {len(labels) - sum(labels)}")
    return texts, labels, metadata_list



# 3. DATASET

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, metadata, tokenizer, max_length, metadata_encoder, metadata_scaler):
        self.texts = texts
        self.labels = labels
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata_encoder = metadata_encoder
        self.metadata_scaler = metadata_scaler

    def __len__(self):
        return len(self.texts)


    def _encode_metadata(self, meta: Dict) -> torch.Tensor:
        party = str(meta.get("political_party", "None"))
        role = str(meta.get("speaker_role", "unknown"))
        country = str(meta.get("country", "UNK"))

        def safe_transform(encoder, value):
            try:
                return encoder.transform([value])[0]
            except ValueError:
                return 0   # unseen label হলে 0 ব্যবহার করবে

        cat_feats = np.array([
            safe_transform(self.metadata_encoder["party"], party),
            safe_transform(self.metadata_encoder["role"], role),
            safe_transform(self.metadata_encoder["country"], country),
        ], dtype=np.float32)

        num_feats = np.array([
            float(meta.get("source_credibility", 0.5)),
            float(meta.get("domain_age", 0))
        ], dtype=np.float32)

        if self.metadata_scaler is not None:
            num_feats = self.metadata_scaler.transform(num_feats.reshape(1, -1)).flatten()

        return torch.tensor(np.concatenate([cat_feats, num_feats]), dtype=torch.float)

   
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["metadata"] = self._encode_metadata(self.metadata[idx])
        return item


def prepare_metadata_encoders(metadata_list):
    parties = [m.get("political_party", "None") for m in metadata_list]
    roles = [m.get("speaker_role", "unknown") for m in metadata_list]
    countries = [m.get("country", "UNK") for m in metadata_list]

    encoders = {
        "party": LabelEncoder().fit(parties),
        "role": LabelEncoder().fit(roles),
        "country": LabelEncoder().fit(countries),
    }

    nums = np.array([
        [float(m.get("source_credibility", 0.5)), float(m.get("domain_age", 0))]
        for m in metadata_list
    ])
    scaler = StandardScaler().fit(nums)
    return encoders, scaler



# 4. MODEL

class MultilingualFakeNewsClassifier(nn.Module):
    def __init__(self, model_name, num_labels, metadata_input_dim, metadata_proj_dim=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_input_dim, metadata_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(metadata_proj_dim, metadata_proj_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + metadata_proj_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, metadata=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        meta_emb = self.metadata_proj(metadata)
        fused = self.fusion(torch.cat([cls_emb, meta_emb], dim=-1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)



# 5. AI MODULE

class TruthForgeExplainer:
    def __init__(self, model, tokenizer, device, metadata_encoder, metadata_scaler):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metadata_encoder = metadata_encoder
        self.metadata_scaler = metadata_scaler
        self.model.eval()

    def _encode_metadata(self, meta: Dict) -> torch.Tensor:
        party = meta.get("political_party", "None")
        role = meta.get("speaker_role", "unknown")
        country = meta.get("country", "UNK")

        cat_feats = np.array([
            self.metadata_encoder["party"].transform([party])[0],
            self.metadata_encoder["role"].transform([role])[0],
            self.metadata_encoder["country"].transform([country])[0],
        ], dtype=np.float32)

        num_feats = np.array([
            float(meta.get("source_credibility", 0.5)),
            float(meta.get("domain_age", 0))
        ], dtype=np.float32)

        if self.metadata_scaler is not None:
            num_feats = self.metadata_scaler.transform(num_feats.reshape(1, -1)).flatten()

        return torch.tensor(np.concatenate([cat_feats, num_feats]), dtype=torch.float)

    def explain_tokens(self, text: str, metadata: Dict, target_class: int = 1):
        """Stable Token Attribution using Input x Gradient method"""
        if not XAI_AVAILABLE:
            return [], 0.0

        self.model.zero_grad()

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=Config.max_length,
            padding="max_length"
        ).to(self.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        meta_tensor = self._encode_metadata(metadata).unsqueeze(0).to(self.device)

        embeddings = self.model.encoder.embeddings(input_ids)
        embeddings.retain_grad()
        embeddings.requires_grad_(True)

        outputs = self.model.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]

        meta_emb = self.model.metadata_proj(meta_tensor)
        fused = self.model.fusion(torch.cat([cls_emb, meta_emb], dim=-1))
        logits = self.model.classifier(fused)

        score = logits[0, target_class]
        score.backward()

        grads = embeddings.grad
        attributions = (embeddings * grads).sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-8)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        scores = attributions.detach().cpu().numpy()

        results = []
        for tok, score in zip(tokens, scores):
            if tok not in self.tokenizer.all_special_tokens:
                results.append((tok, float(score)))

        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)[:12]
        return results, 0.0

    def generate_report(self, text, metadata, pred_label, confidence):
        token_attr, _ = self.explain_tokens(text, metadata, target_class=pred_label)
        return {
            "prediction": "Fake" if pred_label == 1 else "Real",
            "confidence": round(confidence, 4),
            "top_tokens": token_attr
        }



# 6. TRAINING

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0)
    }


def train_model(train_dataset, val_dataset, metadata_input_dim, tokenizer):
    model = MultilingualFakeNewsClassifier(
        Config.model_name, Config.num_labels,
        metadata_input_dim=metadata_input_dim,
        metadata_proj_dim=Config.metadata_dim
    ).to(Config.device)

    training_args = TrainingArguments(
        output_dir="./truthforge_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.lr,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        num_train_epochs=Config.epochs,
        weight_decay=Config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=Config.seed,
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("\nStarting training...")
    trainer.train()
    return model, trainer



# 7. MAIN

if __name__ == "__main__":

    # ---------- Load Data ----------
    if Config.use_real_data:
        print("Loading real data from CSV...")
        texts, labels, metadata_list = load_real_data(Config.real_data_path)
    else:
        print("Creating synthetic dataset...")
        texts, labels, metadata_list = create_synthetic_data(n_samples=80)

    print(f"Total samples: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    encoders, scaler = prepare_metadata_encoders(metadata_list)

    # Get metadata dimension
    temp_ds = FakeNewsDataset(texts[:1], labels[:1], metadata_list[:1],
                              tokenizer, Config.max_length, encoders, scaler)
    meta_dim = temp_ds[0]["metadata"].shape[0]
    print(f"Metadata feature dimension: {meta_dim}")

    # Train-Val split
    train_texts, val_texts, train_labels, val_labels, train_meta, val_meta = train_test_split(
        texts, labels, metadata_list, test_size=0.25, random_state=Config.seed, stratify=labels
    )

    train_ds = FakeNewsDataset(train_texts, train_labels, train_meta,
                               tokenizer, Config.max_length, encoders, scaler)
    val_ds = FakeNewsDataset(val_texts, val_labels, val_meta,
                             tokenizer, Config.max_length, encoders, scaler)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---------- Training ----------
    model, trainer = train_model(train_ds, val_ds, meta_dim, tokenizer)

    # ---------- Inference Demo ----------
    print("\n" + "="*60)
    print("INFERENCE DEMO")
    print("="*60)

    model.eval()
    sample_text = val_texts[0]
    sample_meta = val_meta[0]
    sample_label = val_labels[0]

    encoding = tokenizer(
        sample_text, truncation=True, padding="max_length",
        max_length=Config.max_length, return_tensors="pt"
    ).to(Config.device)

    meta_tensor = train_ds._encode_metadata(sample_meta).unsqueeze(0).to(Config.device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            metadata=meta_tensor
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        confidence = float(probs[pred])

    print(f"Text      : {sample_text}")
    print(f"Metadata  : {sample_meta}")
    print(f"True label: {'Fake' if sample_label == 1 else 'Real'}")
    print(f"Predicted : {'Fake' if pred == 1 else 'Real'} (confidence: {confidence:.3f})")

    # ---------- XAI Explanation ----------
    if XAI_AVAILABLE:
        print("\n" + "="*60)
        print("XAI EXPLANATION")
        print("="*60)

        explainer = TruthForgeExplainer(model, tokenizer, Config.device, encoders, scaler)
        report = explainer.generate_report(sample_text, sample_meta, pred, confidence)

        print(f"Prediction : {report['prediction']} | Confidence: {report['confidence']}")
        print("\nTop influential tokens:")
        for tok, score in report["top_tokens"]:
            print(f"  {tok:20s} -> {score:+.4f}")
    else:
        print("\nTo enable XAI run: pip install captum")



    # Interactive Prediction (Check your own news)
   
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Check Your Own News")
    print("="*60)
    print("Type a news text and press Enter. Type 'exit' to quit.\n")

    while True:
        user_text = input("Enter news: ").strip()
        if user_text.lower() in ["exit", "quit", "q"]:
            break
        if not user_text:
            continue

        # Default metadata
        user_meta = {
            "political_party": "BJP",
            "speaker_role": "politician",
            "country": "IN",
            "source_credibility": 0.5,
            "domain_age": 5
        }

        encoding = tokenizer(
            user_text,
            truncation=True,
            padding="max_length",
            max_length=Config.max_length,
            return_tensors="pt"
        ).to(Config.device)

        meta_tensor = train_ds._encode_metadata(user_meta).unsqueeze(0).to(Config.device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                metadata=meta_tensor
            )
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred = torch.argmax(probs).item()
            confidence = float(probs[pred])

        print(f"\n--> Prediction : {'FAKE' if pred == 1 else 'REAL'}")
        print(f"--> Confidence : {confidence:.3f}")

        if XAI_AVAILABLE:
            report = explainer.generate_report(user_text, user_meta, pred, confidence)
            print("--> Influential words:")
            for tok, score in report["top_tokens"][:6]:
                print(f"     {tok:15s} {score:+.4f}")
        print("-" * 40)

    print("\nScript finished successfully!")