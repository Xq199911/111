#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查模型文件完整性
验证模型文件是否完整且未损坏
"""
import os
import json
from pathlib import Path

def check_model_files(model_path):
    """检查模型文件"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"检查模型: {model_path}")
    print("=" * 60)
    
    # 检查必需文件
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "generation_config.json"
    ]
    
    all_ok = True
    
    # 检查配置文件
    print("\n配置文件:")
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                size = file_path.stat().st_size / 1024  # KB
                print(f"  ✅ {file} ({size:.2f} KB)")
            except Exception as e:
                print(f"  ❌ {file} - 损坏: {e}")
                all_ok = False
        else:
            print(f"  ❌ {file} - 缺失")
            all_ok = False
    
    # 检查模型权重文件
    print("\n模型权重文件:")
    
    # 检查是否有safetensors索引文件
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        print(f"  ✅ model.safetensors.index.json 存在")
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # 检查每个分片
            weight_map = index_data.get("weight_map", {})
            shard_files = set(weight_map.values())
            
            print(f"  需要 {len(shard_files)} 个分片文件")
            for shard_file in sorted(shard_files):
                shard_path = model_path / shard_file
                if shard_path.exists():
                    size_gb = shard_path.stat().st_size / (1024**3)
                    print(f"    ✅ {shard_file} ({size_gb:.2f} GB)")
                    
                    # 尝试打开safetensors文件验证
                    try:
                        from safetensors import safe_open
                        with safe_open(shard_path, framework="pt") as f:
                            keys = list(f.keys())
                            print(f"      包含 {len(keys)} 个tensors")
                    except Exception as e:
                        print(f"    ⚠️  {shard_file} - 可能损坏: {e}")
                        all_ok = False
                else:
                    print(f"    ❌ {shard_file} - 缺失")
                    all_ok = False
        except Exception as e:
            print(f"  ❌ 无法读取索引文件: {e}")
            all_ok = False
    else:
        # 检查是否有单个safetensors文件
        safetensors_file = model_path / "model.safetensors"
        if safetensors_file.exists():
            size_gb = safetensors_file.stat().st_size / (1024**3)
            print(f"  ✅ model.safetensors ({size_gb:.2f} GB)")
            
            # 验证文件
            try:
                from safetensors import safe_open
                with safe_open(safetensors_file, framework="pt") as f:
                    keys = list(f.keys())
                    print(f"    包含 {len(keys)} 个tensors")
            except Exception as e:
                print(f"  ❌ model.safetensors - 损坏: {e}")
                all_ok = False
        else:
            print("  ❌ 未找到模型权重文件")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 模型文件完整性检查通过")
    else:
        print("❌ 模型文件存在问题，需要重新下载")
    
    return all_ok

def main():
    import sys
    
    model_path = "./models/Qwen2.5-3B-Instruct"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("模型完整性检查工具")
    print("=" * 60)
    print()
    
    if not check_model_integrity(model_path):
        print("\n建议:")
        print("1. 重新下载模型文件")
        print("2. 检查磁盘空间是否充足")
        print("3. 检查文件传输是否完整")
        print("\n重新下载命令:")
        print(f"  python download_model.py --model_name Qwen2.5-3B-Instruct --output_dir {os.path.dirname(model_path)}")
        sys.exit(1)
    else:
        print("\n模型文件完整，可以正常使用")
        sys.exit(0)

if __name__ == "__main__":
    main()

