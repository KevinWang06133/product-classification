# 探索数据集结构
from collections import Counter
from modelscope.msdatasets import MsDataset

ds = MsDataset.load('winwin_inc/product-classification-hiring-demo',
                    subset_name='default', split='train')
print("数据集大小:", len(ds))
print("特征列:", ds.column_names)
print("\n前5条样本:")
for i, item in enumerate(ds):
    if i >= 5:
        break
    print(f"  [{i}] {item}")

# 统计类目分布
labels = [item['label'] for item in ds]
counter = Counter(labels)
print("\n类目分布:")
for label, count in counter.most_common():
    print(f"  {label}: {count}")
