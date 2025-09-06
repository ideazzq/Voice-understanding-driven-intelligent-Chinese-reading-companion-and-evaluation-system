import os
import sys
# 找到目录
sys.path.insert(0, r"E:\projects\ChineseReadingTutor\1_asr")

import torch
import librosa
import numpy as np
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from data_collator_ctc import DataCollatorCTCWithPadding

# ---------- 路径 ----------
WAV_DIR   = r"E:\projects\ChineseReadingTutor\data\train"
TXT_DIR   = r"E:\projects\ChineseReadingTutor\data\train"
MODEL_DIR = r"E:\projects\ckpt_asr"          # 刚训好的 ASR 模型
OUTPUT_DIR= r"E:\projects\gop\ckpt_gop"      # GOP 输出目录
SAMPLE_RATE = 16000

processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).eval()

# ---------- 读取 ----------
def read():
    samples = []
    trn_map = {f.lower(): f for f in os.listdir(TXT_DIR)}
    for wav in os.listdir(WAV_DIR):
        if wav.lower().endswith(".wav"):
            trn_key = (wav + ".trn").lower()
            if trn_key in trn_map:
                with open(os.path.join(TXT_DIR, trn_map[trn_key]), encoding="utf-8") as f:
                    text = f.readline().strip()
                samples.append({"audio": os.path.join(WAV_DIR, wav), "text": text})
    return samples

# ---------- GOP 计算 ----------
def gop_score(wav_path, text):
    speech, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with processor.as_target_processor():
        label_ids = processor(text).input_ids

    with torch.no_grad():
        logits = model(**inputs).logits          # [1, T, vocab]
    probs = torch.softmax(logits, dim=-1)
    # GOP：平均最大概率
    score = probs.max(dim=-1).values.mean().item()
    return min(100, int(score * 100))

# ---------- 批量打分 ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = read()
    ds = Dataset.from_list(raw).select(range(1000))
    ds = ds.map(lambda x: {"gop": gop_score(x["audio"], x["text"])}, batched=False)
    ds.save_to_disk(os.path.join(OUTPUT_DIR, "gop_dataset"))
    print(">>> GOP 评分完成，结果已保存到", OUTPUT_DIR)

if __name__ == "__main__":
    main()
