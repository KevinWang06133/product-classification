"""
商品标题分类模型 — RoBERTa 中文预训练微调
==========================================
基于 hfl/chinese-roberta-wwm-ext 进行 10 分类微调
对比 TF-IDF 方案，捕获更深层语义信息
"""

import os
import json
import time
import warnings
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # 中文 RoBERTa (全词掩码)
# 本地 safetensors 快照路径（绕过 torch<2.6 安全限制）
LOCAL_MODEL_PATH = os.path.expanduser(
    r"~/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext"
    r"/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44"
)
MAX_LENGTH = 64                                # 商品标题较短，64 token 足够
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "./BERT/roberta_product_cls"

# 10 个类目
LABEL_LIST = [
    "白酒", "方便面", "牙膏", "口香糖",
    "即饮奶茶", "即饮茶",
    "硬糖", "软糖",
    "冷藏即饮果汁", "非冷藏即饮果汁",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ──────────────────────────────────────────────
# 1. 数据加载
# ──────────────────────────────────────────────
def load_data():
    """从 ModelScope 加载训练集和测试集"""
    from modelscope.msdatasets import MsDataset

    print("正在加载数据集...")
    train_ds = MsDataset.load(
        "winwin_inc/product-classification-hiring-demo",
        subset_name="default", split="train",
    )
    test_ds = MsDataset.load(
        "winwin_inc/product-classification-hiring-demo",
        subset_name="default", split="test",
    )

    train_texts = [item["product_name"] for item in train_ds]
    train_labels = [LABEL2ID[item["category"]] for item in train_ds]
    test_texts = [item["product_name"] for item in test_ds]
    test_labels = [LABEL2ID[item["category"]] for item in test_ds]

    print(f"  训练集: {len(train_texts)} 条")
    print(f"  测试集: {len(test_texts)} 条")
    print(f"  类目数: {len(LABEL_LIST)}")
    return train_texts, train_labels, test_texts, test_labels


# ──────────────────────────────────────────────
# 2. PyTorch Dataset
# ──────────────────────────────────────────────
class ProductDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 3. 评估指标计算
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ──────────────────────────────────────────────
# 4. 训练
# ──────────────────────────────────────────────
def train():
    train_texts, train_labels, test_texts, test_labels = load_data()

    # 从训练集中划分 10% 作为验证集（用于早停）
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels,
    )
    print(
        f"  训练: {len(train_texts)}  验证: {len(val_texts)}  测试: {len(test_texts)}")

    # 加载 tokenizer 和模型
    # 优先从本地 safetensors 快照加载（避免 torch<2.6 的 torch.load 限制）
    model_path = LOCAL_MODEL_PATH if os.path.isdir(
        LOCAL_MODEL_PATH) else MODEL_NAME
    print(f"\n正在加载模型 (from {model_path}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=os.path.isdir(LOCAL_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=os.path.isdir(LOCAL_MODEL_PATH),
    )
    print(f"  模型参数量: {model.num_parameters() / 1e6:.1f}M")

    # 构建 Dataset
    train_dataset = ProductDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProductDataset(val_texts, val_labels, tokenizer)
    test_dataset = ProductDataset(test_texts, test_labels, tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),        # GPU 开启混合精度加速
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始微调训练...")
    print("=" * 60)
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    print(f"\n训练总耗时: {train_time:.1f}s ({train_time / 60:.1f}min)")
    print(
        f"  最佳 epoch 步数: {train_result.metrics.get('train_steps_per_second', 'N/A')}")

    # 保存模型
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"  最佳模型已保存至: {best_model_dir}")

    # ──────────────────────────────────────────
    # 5. 测试集评估
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    # 用最佳模型预测
    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = test_labels

    acc = accuracy_score(true_ids, pred_ids)
    f1_macro = f1_score(true_ids, pred_ids, average="macro")
    f1_weighted = f1_score(true_ids, pred_ids, average="weighted")

    print(f"\n准确率 (Accuracy):  {acc:.4f}")
    print(f"宏平均 F1:          {f1_macro:.4f}")
    print(f"加权 F1:            {f1_weighted:.4f}")

    pred_labels = [ID2LABEL[p] for p in pred_ids]
    true_labels = [ID2LABEL[t] for t in true_ids]

    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels,
          target_names=LABEL_LIST, zero_division=0))

    # 混淆矩阵
    cm = confusion_matrix(true_ids, pred_ids)
    print("混淆矩阵:")
    header = f"{'':>16}" + "".join(f"{lb:>10}" for lb in LABEL_LIST)
    print(f"  {header}")
    for i, lb in enumerate(LABEL_LIST):
        row = f"  {lb:>16}" + \
            "".join(f"{cm[i][j]:>10}" for j in range(len(LABEL_LIST)))
        print(row)

    # 错误分析
    print("\n错误样本 TOP-20:")
    wrong_indices = [i for i in range(
        len(true_ids)) if pred_ids[i] != true_ids[i]]
    for idx in wrong_indices[:20]:
        print(f"  [{idx}] {test_texts[idx]}")
        print(f"       真实: {true_labels[idx]}  预测: {pred_labels[idx]}")

    # ──────────────────────────────────────────
    # 6. 与 TF-IDF 方案对比
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("方案对比汇总")
    print("=" * 60)
    print(f"  TF-IDF (char n-gram) + LinearSVC:  Accuracy=0.9883  F1=0.9841")
    print(
        f"  RoBERTa-wwm-ext 微调:              Accuracy={acc:.4f}  F1={f1_macro:.4f}")

    return acc, f1_macro


if __name__ == "__main__":
    train()
