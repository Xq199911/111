import json
import os
from tqdm import tqdm

# 配置
INPUT_FILE = "./StreamingLLM_GPE/data/inference_data.json"
OUTPUT_FILE = "./StreamingLLM_GPE/data/long_inference_data.json"
TARGET_LENGTH = 1500  # 目标单词数 (我们要超过 1000)
SOURCE_KEY = "en"  # 你的源语言键名
TARGET_KEY = "fr"  # 你的目标语言键名


def create_long_dataset():
    print(f"正在读取原始数据: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始样本数: {len(data)}")

    long_data = []
    current_source = []
    current_target = []
    current_length = 0

    # 循环拼接
    for item in tqdm(data):
        src_text = item[SOURCE_KEY].strip()
        tgt_text = item[TARGET_KEY].strip()

        # 简单的分词估算长度
        word_count = len(src_text.split())

        current_source.append(src_text)
        current_target.append(tgt_text)
        current_length += word_count

        # 如果长度达标，就保存为一个新样本
        if current_length >= TARGET_LENGTH:
            new_entry = {
                SOURCE_KEY: " ".join(current_source),
                TARGET_KEY: " ".join(current_target)
            }
            long_data.append(new_entry)

            # 重置缓冲区
            current_source = []
            current_target = []
            current_length = 0

    # 如果最后还有剩余，也算作一个样本（只要不太短）
    if current_length > 500:
        new_entry = {
            SOURCE_KEY: " ".join(current_source),
            TARGET_KEY: " ".join(current_target)
        }
        long_data.append(new_entry)

    print(f"\n生成完毕!")
    print(f"新数据集样本数: {len(long_data)}")
    print(f"输出文件: {OUTPUT_FILE}")

    # 验证第一个样本长度
    if len(long_data) > 0:
        first_len = len(long_data[0][SOURCE_KEY].split())
        print(f"样例 0 长度: {first_len} words (符合 > 1000 的要求)")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(long_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    create_long_dataset()