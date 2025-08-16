import os, json, math, re
from dataclasses import dataclass
from typing import Dict, List
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
import argparse, yaml

@dataclass
class Prompter:
    template: str
    input_key: str
    response_key: str
    def build(self, ex: Dict) -> str:
        inp = ex.get(self.input_key, "") or ""
        return self.template.format(input=inp)

def load_config(path: str) -> dict:
    sci_pattern = re.compile(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$')
    def auto_cast(val):
        if isinstance(val, str) and sci_pattern.match(val):
            try:
                return float(val)
            except ValueError:
                return val
        elif isinstance(val, dict):
            return {k: auto_cast(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [auto_cast(x) for x in val]
        return val
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)    
    return auto_cast(cfg)

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    model_name = cfg["model_name"]
    dataset_id = cfg["dataset"]
    text_col = cfg.get("text_column", "instruction")
    resp_col = cfg.get("response_column", "output")
    template = cfg.get("prompt_template", "{input}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompter = Prompter(template, text_col, resp_col)

    ds = load_dataset(dataset_id)
    def to_text(ex):
        prompt = prompter.build(ex)
        resp = ex.get(resp_col, "") or ""
        full = prompt + (resp if resp.endswith(tokenizer.eos_token) else resp + tokenizer.eos_token)
        return {"text": full}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)
    ds = ds.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    # 让 pad_token_id 与 eos 对齐，避免警告
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 2e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=cfg.get("bf16", False),
        tf32=cfg.get("tf32", True),
        report_to=cfg.get("report_to", "none"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 断点续训：自动找到最近的 checkpoint
    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        ckpts = [os.path.join(training_args.output_dir, d)
                 for d in os.listdir(training_args.output_dir)
                 if d.startswith("checkpoint-")]
        if ckpts:
            last_ckpt = sorted(ckpts, key=lambda p: int(p.split("-")[-1]))[-1]

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(training_args.output_dir)    # 保存最终权重（含adapter时也会保存）

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()
    main(args.config)

# python code/train_sft.py --config configs/train.yaml