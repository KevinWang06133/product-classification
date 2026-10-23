"""将缓存的 pytorch_model.bin 转为 safetensors 格式"""
import os
import glob
import torch
from safetensors.torch import save_file

# 查找 transformers 缓存中的 pytorch_model.bin
cache_pattern = os.path.expanduser(
    "~/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext/snapshots/*/pytorch_model.bin"
)
candidates = glob.glob(cache_pattern)

if not candidates:
    print("缓存中未找到 pytorch_model.bin，尝试从 ModelScope 缓存查找...")
    # ModelScope 可能使用不同的缓存路径
    cache_pattern2 = os.path.expanduser(
        "~/.cache/modelscope/hub/models/hfl/chinese-roberta-wwm-ext/*.bin"
    )
    candidates = glob.glob(cache_pattern2)

if not candidates:
    print("未找到缓存文件，尝试直接下载并转换...")
    # 直接用 torch.load 从 HuggingFace 缓存加载
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("hfl/chinese-roberta-wwm-ext", "pytorch_model.bin")
    candidates = [path]

bin_path = candidates[0]
print(f"找到: {bin_path}")
print(f"文件大小: {os.path.getsize(bin_path) / 1024**2:.0f} MB")

print("正在加载 state_dict...")
state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

# 解决共享内存张量 + 非连续张量问题
print("正在处理张量（克隆共享 + 连续化）...")
seen_data_ptrs = {}
for key, tensor in list(state_dict.items()):
    ptr = tensor.data_ptr()
    if ptr in seen_data_ptrs:
        state_dict[key] = tensor.clone().contiguous()
        print(f"  克隆: {key} (与 {seen_data_ptrs[ptr]} 共享内存)")
    else:
        seen_data_ptrs[ptr] = key
        if not tensor.is_contiguous():
            state_dict[key] = tensor.contiguous()
            print(f"  连续化: {key}")

safetensor_path = bin_path.replace("pytorch_model.bin", "model.safetensors")
print(f"正在保存 safetensors -> {safetensor_path}")
save_file(state_dict, safetensor_path)
print(f"完成! 文件大小: {os.path.getsize(safetensor_path) / 1024**2:.0f} MB")
