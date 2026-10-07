"""
TruthForge v2 - Multilingual fake news detection (text + metadata) with explainability.

Install:  pip install torch transformers scikit-learn pandas numpy captum
Data   :  truthforge_data.json -> [{"text": ..., "label": 0/1, "metadata": {...}}, ...]
"""
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoTokenizer, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from captum.attr import LayerIntegratedGradients
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False


# ----------------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # Stronger options: "xlm-roberta-large", "microsoft/mdeberta-v3-base"
    # Fast testing:     "distilbert-base-multilingual-cased"
    model_name: str = "xlm-roberta-base"
    data_path: str = "truthforge_data.json"
    save_path: str = "truthforge_model.pt"

    max_length: int = 256          # news is longer than 128 tokens
    batch_size: int = 16
    grad_accum: int = 2            # effective batch = 32
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.2

    use_metadata: bool = True      # set False for the text-only ablation
    cat_emb_dim: int = 8
    meta_dim: int = 32
    meta_dropout: float = 0.2      # randomly hide metadata so the model can't rely on it alone

    num_labels: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


LABELS = ["Real", "Fake"]


# ----------------------------------------------------------------------------
# 2. DATA
# ----------------------------------------------------------------------------
CAT_FIELDS = {"political_party": "None", "speaker_role": "unknown", "country": "UNK"}
NUM_FIELDS = {"source_credibility": 0.5, "domain_age": 0.0}


def load_data(path: str) -> Tuple[List[str], List[int], List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [str(d["text"]) for d in data]
    labels = [int(d["label"]) for d in data]
    metas = [d.get("metadata", {}) for d in data]
    print(f"Loaded {len(texts)} samples | fake={sum(labels)} real={len(labels) - sum(labels)}")
    return texts, labels, metas


class MetadataProcessor:
    """Fit on TRAIN data only. Index 0 of every vocab is <UNK> (unseen values)."""

    def fit(self, metas: List[Dict]) -> "MetadataProcessor":
        self.vocab = {
            field: {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(
                sorted({str(m.get(field, default)) for m in metas}))}}
            for field, default in CAT_FIELDS.items()
        }
        nums = self._nums(metas)
        self.mean, self.std = nums.mean(0), nums.std(0) + 1e-6
        return self

    @staticmethod
    def _nums(metas: List[Dict]) -> np.ndarray:
        return np.array([[float(m.get(f, d)) for f, d in NUM_FIELDS.items()] for m in metas],
                        dtype=np.float32)

    @property
    def vocab_sizes(self) -> List[int]:
        return [len(v) for v in self.vocab.values()]

    def transform(self, metas: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = [[self.vocab[f].get(str(m.get(f, d)), 0) for f, d in CAT_FIELDS.items()] for m in metas]
        num = (self._nums(metas) - self.mean) / self.std
        return torch.tensor(cat, dtype=torch.long), torch.tensor(num, dtype=torch.float)

    def to_dict(self) -> Dict:
        return {"vocab": self.vocab, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "MetadataProcessor":
        p = cls()
        p.vocab, p.mean, p.std = d["vocab"], np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
        return p


class NewsDataset(Dataset):
    """Tokenizes once up-front; padding is done per batch (dynamic) in the collator."""

    def __init__(self, texts, labels, metas, tokenizer, processor, max_length):
        self.ids = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.cat, self.num = processor.transform(metas)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i],
                "meta_cat": self.cat[i], "meta_num": self.num[i]}


def make_collator(tokenizer):
    def collate(batch):
        enc = tokenizer.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
        enc["labels"] = torch.stack([b["labels"] for b in batch])
        enc["meta_cat"] = torch.stack([b["meta_cat"] for b in batch])
        enc["meta_num"] = torch.stack([b["meta_num"] for b in batch])
        return dict(enc)
    return collate


# ----------------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------------
def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


class FakeNewsClassifier(nn.Module):
    def __init__(self, cfg: Config, vocab_sizes: List[int], class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.encoder.config.hidden_size

        meta_out = 0
        if cfg.use_metadata:
            self.cat_emb = nn.ModuleList([nn.Embedding(n, cfg.cat_emb_dim) for n in vocab_sizes])
            meta_in = len(vocab_sizes) * cfg.cat_emb_dim + len(NUM_FIELDS)
            self.meta_mlp = nn.Sequential(nn.Linear(meta_in, cfg.meta_dim), nn.GELU(), nn.Dropout(0.1))
            meta_out = cfg.meta_dim

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden + meta_out, hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.num_labels),
        )
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids=None, attention_mask=None, meta_cat=None, meta_num=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        feats = mean_pool(h, attention_mask)

        if self.cfg.use_metadata:
            emb = torch.cat([layer(meta_cat[:, i]) for i, layer in enumerate(self.cat_emb)] + [meta_num], dim=-1)
            meta = self.meta_mlp(emb)
            if self.training and self.cfg.meta_dropout > 0:   # modality dropout
                keep = (torch.rand(meta.size(0), 1, device=meta.device) > self.cfg.meta_dropout).to(meta.dtype)
                meta = meta * keep
            feats = torch.cat([feats, meta], dim=-1)

        logits = self.head(feats)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ----------------------------------------------------------------------------
# 4. TRAINING
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1, "precision": p, "recall": r}


def train(cfg, model, train_ds, val_ds, tokenizer):
    # warmup_ratio was removed in newer transformers versions -> compute steps ourselves (works in all versions)
    steps_per_epoch = -(-len(train_ds) // (cfg.batch_size * cfg.grad_accum))
    warmup_steps = int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs)

    args = TrainingArguments(
        output_dir="./truthforge_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
        logging_steps=20,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=make_collator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    return trainer


def evaluate_on_test(trainer, test_ds):
    out = trainer.predict(test_ds)
    preds = out.predictions.argmax(-1)
    print("\nTEST RESULTS")
    print(classification_report(out.label_ids, preds, target_names=LABELS, digits=4))


# ----------------------------------------------------------------------------
# 5. SAVE / LOAD
# ----------------------------------------------------------------------------
def save_artifacts(cfg, model, processor):
    torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict(),
                "processor": processor.to_dict()}, cfg.save_path)
    print(f"Saved to {cfg.save_path}")


def load_artifacts(path: str):
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["cfg"])
    processor = MetadataProcessor.from_dict(ckpt["processor"])
    model = FakeNewsClassifier(cfg, processor.vocab_sizes)
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return cfg, model.to(cfg.device).eval(), tokenizer, processor


# ----------------------------------------------------------------------------
# 6. PREDICTION + EXPLANATION
# ----------------------------------------------------------------------------
class Predictor:
    def __init__(self, cfg, model, tokenizer, processor):
        self.cfg, self.model, self.tokenizer, self.processor = cfg, model.eval(), tokenizer, processor
        self.special = set(tokenizer.all_special_tokens)
        self.lig = None
        if XAI_AVAILABLE:
            self.lig = LayerIntegratedGradients(self._logits, model.encoder.embeddings.word_embeddings)

    def _inputs(self, text: str, meta: Optional[Dict]):
        enc = self.tokenizer(text, truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
        cat, num = self.processor.transform([meta or {}])   # missing metadata -> neutral defaults
        return (enc["input_ids"].to(self.cfg.device), enc["attention_mask"].to(self.cfg.device),
                cat.to(self.cfg.device), num.to(self.cfg.device))

    def _logits(self, input_ids, attention_mask, meta_cat, meta_num):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          meta_cat=meta_cat, meta_num=meta_num).logits

    @torch.no_grad()
    def predict(self, text: str, meta: Optional[Dict] = None) -> Dict:
        probs = F.softmax(self._logits(*self._inputs(text, meta)), dim=-1)[0].cpu()
        pred = int(probs.argmax())
        return {"label": LABELS[pred], "pred": pred, "confidence": float(probs[pred]),
                "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)}}

    def explain(self, text: str, meta: Optional[Dict] = None, target: Optional[int] = None,
                top_k: int = 10, n_steps: int = 32) -> List[Tuple[str, float]]:
        """Integrated Gradients over word embeddings. Positive score = pushes toward `target`."""
        if self.lig is None:
            raise RuntimeError("captum not installed: pip install captum")
        if target is None:
            target = self.predict(text, meta)["pred"]

        ids, mask, cat, num = self._inputs(text, meta)
        baseline = torch.full_like(ids, self.tokenizer.pad_token_id)
        baseline[:, 0], baseline[:, -1] = ids[:, 0], ids[:, -1]       # keep <s> and </s>

        attr = self.lig.attribute(ids, baselines=baseline, additional_forward_args=(mask, cat, num),
                                  target=target, n_steps=n_steps)
        scores = attr.sum(-1).squeeze(0)
        scores = (scores / (scores.norm() + 1e-8)).detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
        return self._merge_subwords(tokens, scores)[:top_k]

    def _merge_subwords(self, tokens, scores):
        words: List[List] = []
        for tok, s in zip(tokens, scores):
            if tok in self.special:
                continue
            if tok.startswith("\u2581") or not words:        # sentencepiece word start
                words.append([tok.lstrip("\u2581"), s])
            else:
                words[-1][0] += tok
                words[-1][1] += s
        words = [(w, s) for w, s in words if w]
        return sorted(words, key=lambda x: abs(x[1]), reverse=True)


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    print(f"Device: {cfg.device} | Model: {cfg.model_name} | metadata: {cfg.use_metadata}")

    texts, labels, metas = load_data(cfg.data_path)

    # 70 / 15 / 15 stratified split
    idx = np.arange(len(texts))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=cfg.seed, stratify=labels)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=cfg.seed,
                                      stratify=[labels[i] for i in tmp_idx])
    pick = lambda ix: ([texts[i] for i in ix], [labels[i] for i in ix], [metas[i] for i in ix])
    (tr_t, tr_l, tr_m), (va_t, va_l, va_m), (te_t, te_l, te_m) = pick(tr_idx), pick(va_idx), pick(te_idx)
    print(f"Train {len(tr_t)} | Val {len(va_t)} | Test {len(te_t)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    processor = MetadataProcessor().fit(tr_m)            # fit on train only -> no leakage

    train_ds = NewsDataset(tr_t, tr_l, tr_m, tokenizer, processor, cfg.max_length)
    val_ds = NewsDataset(va_t, va_l, va_m, tokenizer, processor, cfg.max_length)
    test_ds = NewsDataset(te_t, te_l, te_m, tokenizer, processor, cfg.max_length)

    counts = np.bincount(tr_l, minlength=cfg.num_labels)
    class_weights = torch.tensor(len(tr_l) / (cfg.num_labels * counts), dtype=torch.float)

    model = FakeNewsClassifier(cfg, processor.vocab_sizes, class_weights).to(cfg.device)
    trainer = train(cfg, model, train_ds, val_ds, tokenizer)
    evaluate_on_test(trainer, test_ds)
    save_artifacts(cfg, trainer.model, processor)

    # ---- Demo + interactive mode ----
    predictor = Predictor(cfg, trainer.model, tokenizer, processor)
    print("\nINTERACTIVE MODE - type news text ('exit' to quit)")
    while True:
        text = input("\nEnter news: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        res = predictor.predict(text)           # no metadata -> neutral defaults
        print(f"--> {res['label'].upper()} (confidence {res['confidence']:.3f})")
        if XAI_AVAILABLE:
            print("--> Influential words (+ supports the prediction, - opposes):")
            for word, score in predictor.explain(text, target=res["pred"], top_k=8):
                print(f"     {word:18s} {score:+.3f}")


if __name__ == "__main__":
    main()
