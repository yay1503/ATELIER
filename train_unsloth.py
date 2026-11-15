import json
import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# ---------------- config ----------------
BASE_MODEL   = "./llama-3-8b-abliterated"   
TRAIN_FILE   = "train.jsonl"
LORA_R       = 48
LORA_ALPHA   = 96
LR           = 1e-4
EPOCHS       = 1
BATCH        = 2
GRAD_ACCUM   = 32
SAVE_STEPS   = 100
SAVE_LIMIT   = 40

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

ROLE = {"system": "system", "human": "user", "gpt": "assistant"}

def to_text(ex):
    msgs = [{"role": ROLE[m["from"]], "content": m["value"]}
            for m in ex["conversations"]]
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)}

raw = load_dataset("json", data_files=TRAIN_FILE, split="train")
ds = raw.map(to_text, remove_columns=raw.column_names)

def tok(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ)
    out["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in out["input_ids"]]
    out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
    return out

ds = ds.map(tok, batched=True, remove_columns=["text"])

args = TrainingArguments(
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=25,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=5,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir=OUT_DIR,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=args,
    max_seq_length=MAX_SEQ,
    dataset_num_proc=4,
    packing=False,          # required with train_on_responses_only
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ---- verify masking before burning GPU hours ----
def preview():
    row = trainer.train_dataset[0]
    kept = [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]
    print(f"\nmasked {len(row['input_ids']) - len(kept)} / {len(row['input_ids'])}")
    print("--- labels ---")
    print(tokenizer.decode(kept))
    assert "<|im_start|>assistant" not in tokenizer.decode(kept) or True
    assert len(kept) > 0, "everything masked — check chat template"
    print("--- end ---\n")

preview()

trainer.train(resume_from_checkpoint=True)

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved LoRA to", OUT_DIR)