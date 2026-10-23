"""
商品标题分类模型
================
基于 TF-IDF + LinearSVC 的商品标题分类器
支持 10 个类目：即饮奶茶、即饮茶、硬糖、牙膏、非冷藏即饮果汁、
              冷藏即饮果汁、白酒、方便面、软糖、口香糖
"""

import json
import time
import jieba
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ──────────────────────────────────────────────
# 1. 数据加载
# ──────────────────────────────────────────────


def load_data():
    """从 ModelScope 加载训练集和测试集"""
    from modelscope.msdatasets import MsDataset

    print("正在加载数据集...")
    train_ds = MsDataset.load(
        "winwin_inc/product-classification-hiring-demo",
        subset_name="default",
        split="train",
    )
    test_ds = MsDataset.load(
        "winwin_inc/product-classification-hiring-demo",
        subset_name="default",
        split="test",
    )

    train_texts = [item["product_name"] for item in train_ds]
    train_labels = [item["category"] for item in train_ds]
    test_texts = [item["product_name"] for item in test_ds]
    test_labels = [item["category"] for item in test_ds]

    print(f"  训练集: {len(train_texts)} 条")
    print(f"  测试集: {len(test_texts)} 条")
    print(f"  类目数: {len(set(train_labels))}")
    print(f"  类目列表: {sorted(set(train_labels))}")

    return train_texts, train_labels, test_texts, test_labels


# ──────────────────────────────────────────────
# 2. 文本预处理
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    中文商品标题预处理：
    - jieba 分词
    - 保留中文字符、字母、数字
    - 用空格连接词语
    """
    words = jieba.lcut(text)
    return " ".join(words)


# ──────────────────────────────────────────────
# 3. 构建分类 Pipeline
# ──────────────────────────────────────────────
def build_pipeline():
    """
    TF-IDF + LinearSVC 分类 Pipeline

    设计思路：
    - TF-IDF 使用 char_wb（字级 n-gram）+ 词级 n-gram 双通道特征
    - LinearSVC 适合高维稀疏特征，训练速度快
    - CalibratedClassifierCV 增加概率输出能力
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",     # 字级 n-gram，对中文无需分词即可捕获局部模式
            ngram_range=(1, 4),     # 1~4 字符片段，覆盖品牌名、品类词等
            max_features=200000,    # 特征上限
            sublinear_tf=True,      # TF 对数缩放，抑制高频字影响
            min_df=2,               # 至少出现 2 次
            max_df=0.95,            # 至多 95% 文档出现
        )),
        ("clf", LinearSVC(
            C=10.0,                 # 正则化强度
            max_iter=10000,
            class_weight="balanced",
        )),
    ])
    return pipeline


# ──────────────────────────────────────────────
# 4. 训练与评估
# ──────────────────────────────────────────────
def train_and_evaluate():
    # 加载数据
    train_texts, train_labels, test_texts, test_labels = load_data()

    # 预处理
    print("\n正在预处理文本（jieba 分词）...")
    t0 = time.time()
    train_processed = [preprocess(t) for t in train_texts]
    test_processed = [preprocess(t) for t in test_texts]
    print(f"  预处理耗时: {time.time() - t0:.1f}s")

    # 同时也构建字级特征（不做分词）
    train_char = train_texts
    test_char = test_texts

    # ── 方案 A：字级 n-gram ──
    print("\n" + "=" * 60)
    print("方案 A：TF-IDF (char n-gram) + LinearSVC")
    print("=" * 60)
    pipeline_char = build_pipeline()

    t0 = time.time()
    pipeline_char.fit(train_char, train_labels)
    print(f"训练耗时: {time.time() - t0:.1f}s")

    pred_char = pipeline_char.predict(test_char)
    acc_char = accuracy_score(test_labels, pred_char)
    f1_char = f1_score(test_labels, pred_char, average="macro")
    print(f"\n准确率 (Accuracy):  {acc_char:.4f}")
    print(f"宏平均 F1:          {f1_char:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, pred_char, zero_division=0))

    # ── 方案 B：词级 n-gram（jieba 分词后）──
    print("=" * 60)
    print("方案 B：TF-IDF (word n-gram, jieba) + LinearSVC")
    print("=" * 60)
    pipeline_word = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=200000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LinearSVC(
            C=10.0,
            max_iter=10000,
            class_weight="balanced",
        )),
    ])

    t0 = time.time()
    pipeline_word.fit(train_processed, train_labels)
    print(f"训练耗时: {time.time() - t0:.1f}s")

    pred_word = pipeline_word.predict(test_processed)
    acc_word = accuracy_score(test_labels, pred_word)
    f1_word = f1_score(test_labels, pred_word, average="macro")
    print(f"\n准确率 (Accuracy):  {acc_word:.4f}")
    print(f"宏平均 F1:          {f1_word:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, pred_word, zero_division=0))

    # ── 方案 C：融合（字级 + 词级特征拼接）──
    print("=" * 60)
    print("方案 C：特征融合 (char + word) + LinearSVC")
    print("=" * 60)
    from scipy.sparse import hstack

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 4),
        max_features=150000, sublinear_tf=True, min_df=2, max_df=0.95,
    )
    tfidf_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        max_features=150000, sublinear_tf=True, min_df=2, max_df=0.95,
    )

    X_train_char = tfidf_char.fit_transform(train_char)
    X_train_word = tfidf_word.fit_transform(train_processed)
    X_train = hstack([X_train_char, X_train_word])

    X_test_char = tfidf_char.transform(test_char)
    X_test_word = tfidf_word.transform(test_processed)
    X_test = hstack([X_test_char, X_test_word])

    clf = LinearSVC(C=10.0, max_iter=10000, class_weight="balanced")
    t0 = time.time()
    clf.fit(X_train, train_labels)
    print(f"训练耗时: {time.time() - t0:.1f}s")

    pred_fusion = clf.predict(X_test)
    acc_fusion = accuracy_score(test_labels, pred_fusion)
    f1_fusion = f1_score(test_labels, pred_fusion, average="macro")
    print(f"\n准确率 (Accuracy):  {acc_fusion:.4f}")
    print(f"宏平均 F1:          {f1_fusion:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, pred_fusion, zero_division=0))

    # ── 汇总对比 ──
    print("\n" + "=" * 60)
    print("方案对比汇总")
    print("=" * 60)
    print(f"  方案 A (字级 n-gram):     Accuracy={acc_char:.4f}  F1={f1_char:.4f}")
    print(f"  方案 B (词级 jieba):      Accuracy={acc_word:.4f}  F1={f1_word:.4f}")
    print(
        f"  方案 C (融合 char+word):  Accuracy={acc_fusion:.4f}  F1={f1_fusion:.4f}")

    best_name, best_acc, best_f1, best_pred = max(
        [("A", acc_char, f1_char, pred_char),
         ("B", acc_word, f1_word, pred_word),
         ("C", acc_fusion, f1_fusion, pred_fusion)],
        key=lambda x: x[2],
    )
    print(f"\n  最优方案: {best_name}  (F1={best_f1:.4f})")

    # ── 混淆矩阵 ──
    labels_sorted = sorted(set(test_labels))
    cm = confusion_matrix(test_labels, best_pred, labels=labels_sorted)
    print(f"\n  混淆矩阵 (方案 {best_name}):")
    print(f"  {'':>14}", end="")
    for lb in labels_sorted:
        print(f"  {lb:>8}", end="")
    print()
    for i, lb in enumerate(labels_sorted):
        print(f"  {lb:>14}", end="")
        for j in range(len(labels_sorted)):
            print(f"  {cm[i][j]:>8}", end="")
        print()

    # ── 错误分析 ──
    print(f"\n  方案 {best_name} 错误样本 TOP-20:")
    wrong_indices = [i for i in range(
        len(test_labels)) if best_pred[i] != test_labels[i]]
    for idx in wrong_indices[:20]:
        print(f"    [{idx}] 标题: {test_texts[idx]}")
        print(f"         真实: {test_labels[idx]}  预测: {best_pred[idx]}")

    return best_name, best_acc, best_f1


if __name__ == "__main__":
    train_and_evaluate()
