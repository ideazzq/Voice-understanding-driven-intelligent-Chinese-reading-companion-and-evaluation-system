import os
import torch
from datasets import load_dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments

OUTPUT_DIR = r"E:\projects\rc\ckpt_rc"
PRETRAINED = "bert-base-chinese"

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
model = BertForQuestionAnswering.from_pretrained(PRETRAINED)

# ---------- 数据：CMRC2018 ----------
raw_ds = load_dataset("cmrc2018")
train_ds = raw_ds["train"].select(range(8000))  # 演示 8k
eval_ds  = raw_ds["validation"]

# ---------- 编码：使用 overflow/sliding window 并将 char idx 转 token idx ----------

# - overflow_to_sample_mapping 告诉我们每个 feature 来自于哪个原始样本索引
def prepare_train_features(examples):
    # examples: batch dict（batched=True）
    # 1) 使用 tokenizer 生成 features（包含 overflow）
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 2) 记录 feature -> original example 关系
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        # 找到原始样本下标
        sample_index = sample_mapping[i]
        # 对应的答案结构可能是两种形式：
        # (A) answers 是 dict: {"text": [...], "answer_start": [...]}
        # (B) answers 是 list of dicts: [{"text": "...", "answer_start": ...}, ...]
        answers = examples["answers"][sample_index]

        # 提取第一个答案（若没有答案，标记为 no-answer）
        if (isinstance(answers, dict) and len(answers.get("text", [])) == 0) or (isinstance(answers, list) and len(answers) == 0):
            # no answer for this example
            start_positions.append(0)
            end_positions.append(0)
            continue

        # 取得字符级的起止（start_char, end_char)
        if isinstance(answers, dict):
            # format: {"text": [...], "answer_start": [...]}
            # 取第一个答案
            answer_text = answers["text"][0]
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answer_text)
        elif isinstance(answers, list):
            # format: list of dicts
            ans0 = answers[0]
            answer_text = ans0.get("text", "")
            start_char = ans0.get("answer_start", 0)
            end_char = start_char + len(answer_text)
        else:
            # 兜底
            start_positions.append(0)
            end_positions.append(0)
            continue

        # 找到 feature 中对应 token 的起始和结束位置
        # offsets: list of (char_start, char_end) for each token in this feature
        token_start_index = 0
        token_end_index = len(offsets) - 1

        # 移动 token_start_index 到第一个 offset 包含 start_char 的 token
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= 0:
            token_start_index += 1
        # 更稳健搜索：从头开始寻找第一个包含 start_char 的 token
        found_start = False
        for idx, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                token_start_index = idx
                found_start = True
                break
        if not found_start:
            # 若答案不在当前 feature（被截断），将位置置为 0 (CLS token)
            start_positions.append(0)
            end_positions.append(0)
            continue

        # 找 token_end_index：第一个 token 的 end >= end_char - 1
        found_end = False
        for idx in range(len(offsets)-1, -1, -1):
            s, e = offsets[idx]
            if s < end_char <= e:
                token_end_index = idx
                found_end = True
                break
        if not found_end:
            # 如果没有找到合适结束 token，也把这个 feature 当作 no-answer
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    # 将 start_positions / end_positions 添加到 tokenized 输出中
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


# 使用 batched map
train_ds = train_ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_ds.column_names,
    load_from_cache_file=False,
)

eval_ds = eval_ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=eval_ds.column_names,
    load_from_cache_file=False,
)

# ---------- 训练参数 ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="no",
    save_strategy="no",
    logging_steps=100,
    learning_rate=3e-5,
    remove_unused_columns=True,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

def main():
    print(">>> 开始阅读理解训练（CMRC2018）... 训练样本:", len(train_ds), "验证样本:", len(eval_ds))
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(">>> 阅读理解训练完成，模型已保存到", OUTPUT_DIR)

if __name__ == "__main__":
    main()
