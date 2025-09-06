# eval_rc.py
import os
import json
import numpy as np
from datasets import load_dataset
import evaluate
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer

MODEL_DIR = r"E:\projects\rc\ckpt_rc"   # 训练输出目录（保存的模型）
PRETRAINED = MODEL_DIR
MAX_ANSWER_LENGTH = 30
DOC_STRIDE = 128
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# load model + tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
model = BertForQuestionAnswering.from_pretrained(PRETRAINED).to(DEVICE)

# load dataset (validation set)
raw_valid = load_dataset("cmrc2018")["validation"]

# tokenization / feature building (same as training)
def prepare_validation_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    # keep track of mapping for postprocessing
    tokenized["example_id"] = []
    tokenized["offset_mapping"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index] if "id" in examples else str(sample_index))
        # we will need offsets to map tokens -> chars; keep them but mark question tokens with (0,0)
        tokenized["offset_mapping"].append([
            (o[0], o[1]) if tokenized["token_type_ids"][i][j] == 1 else (0, 0)
            for j, o in enumerate(offsets)
        ])

    return tokenized

print("Tokenizing validation set (this may take a while)...")
features_valid = raw_valid.map(
    prepare_validation_features,
    batched=True,
    remove_columns=raw_valid.column_names,
    load_from_cache_file=False
)

# predict with Trainer.predict (returns tuple / np arrays)
trainer = Trainer(model=model)

print("Running predictions (this may take a while)...")
predictions = trainer.predict(features_valid)
# predictions.predictions may be tuple (start_logits, end_logits) or array (n,2,m)
preds = predictions.predictions
if isinstance(preds, tuple) or (isinstance(preds, list) and len(preds) == 2):
    start_logits = preds[0]
    end_logits = preds[1]
else:
    # sometimes shape is (N, seq_len, 2)
    if preds.ndim == 3 and preds.shape[-1] == 2:
        start_logits = preds[..., 0]
        end_logits = preds[..., 1]
    else:
        raise ValueError("无法解析模型输出形状：" + str(preds.shape))

# Post-process: convert logits + offsets -> final text predictions
def postprocess_qa_predictions(examples, features, start_logits, end_logits, n_best_size=20, max_answer_length=MAX_ANSWER_LENGTH):
    example_id_to_index = {str(eid): i for i, eid in enumerate(examples["id"]) if "id" in examples}
    features_per_example = {}
    for i, feat in enumerate(features):
        eid = str(feat["example_id"])
        features_per_example.setdefault(eid, []).append(i)

    predictions = {}

    for eid, feature_idxs in features_per_example.items():
        example_index = example_id_to_index.get(eid, None)
        if example_index is None:
            # fallback: use numeric index if ids absent
            example_index = int(eid)

        context = examples["context"][example_index]
        prelim = []

        for fi in feature_idxs:
            offsets = features[fi]["offset_mapping"]
            s_logits = start_logits[fi]
            e_logits = end_logits[fi]

            # get top n start and end indices
            start_indexes = np.argsort(s_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(e_logits)[-1: -n_best_size - 1: -1].tolist()

            for si in start_indexes:
                for ei in end_indexes:
                    if si >= len(offsets) or ei >= len(offsets):
                        continue
                    # only consider positions mapping to context (offset != (0,0))
                    if offsets[si] == (0, 0) or offsets[ei] == (0, 0):
                        continue
                    if ei < si:
                        continue
                    length = ei - si + 1
                    if length > max_answer_length:
                        continue
                    start_char = offsets[si][0]
                    end_char = offsets[ei][1]
                    score = s_logits[si] + e_logits[ei]
                    prelim.append({
                        "score": float(score),
                        "start": int(start_char),
                        "end": int(end_char),
                        "text": context[start_char:end_char]
                    })
        if len(prelim) == 0:
            # no valid span, output empty string
            predictions[eid] = ""
        else:
            best = sorted(prelim, key=lambda x: x["score"], reverse=True)[0]
            predictions[eid] = best["text"]
    return predictions

print("Post-processing predictions...")
# HF Dataset maps store as Arrow Tables; convert to plain python lists/dicts for postprocessing
raw_valid_small = raw_valid  # contains 'context', 'id', etc.
# convert features_valid to list of dicts
feat_list = []
for f in features_valid:
    # ensure token_type_ids accessible
    feat_list.append({
        "input_ids": f["input_ids"],
        "token_type_ids": f["token_type_ids"],
        "attention_mask": f["attention_mask"],
        "offset_mapping": f["offset_mapping"],
        "example_id": f["example_id"]
    })

preds_dict = postprocess_qa_predictions(raw_valid_small, feat_list, start_logits, end_logits)

# Prepare metric format
preds_list = [{"id": eid, "prediction_text": text} for eid, text in preds_dict.items()]
refs = []
for i, ex in enumerate(raw_valid_small):
    eid = str(ex["id"]) if "id" in ex else str(i)
    # construct reference: prefer first answer
    answers = ex["answers"]
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, list):
        texts = [a.get("text", "") for a in answers]
    else:
        texts = []
    refs.append({"id": eid, "answers": {"text": texts, "answer_start": ex["answers"].get("answer_start", []) if isinstance(ex["answers"], dict) else [a.get("answer_start", 0) for a in ex["answers"]] if isinstance(ex["answers"], list) else []}})

# compute squad metrics (EM, F1)
squad_metric = evaluate.load("squad")
results = squad_metric.compute(predictions=preds_list, references=refs)
print("Evaluation results:", results)

# save predictions
out_path = os.path.join(MODEL_DIR, "cmrc_valid_predictions.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(preds_list, f, ensure_ascii=False, indent=2)
print("Saved predictions to", out_path)
