import torch
import transformers
from datasets import load_dataset
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse

def print_trainable_parameters(model):
    """学習可能なパラメータ数と割合を表示"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main(args):
    # --- 1. QLoRA設定 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. LoRA設定 ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA設定後のモデル:")
    print_trainable_parameters(model)

    # --- 3. データセット ---
    dataset = load_dataset(args.dataset_name, "med_qa_en_bigbio_qa", split="train", trust_remote_code=True)

    def format_data(example):
        options = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(example["choices"])])
        answer_text = example["answer"][0] if isinstance(example["answer"], list) else example["answer"]
        prompt = f"Question: {example['question']}\n\nOptions:\n{options}\n\nAnswer:"
        return {"prompt": prompt, "label": " " + answer_text}

    dataset = dataset.map(format_data, remove_columns=[
        'id', 'question_id', 'document_id', 'question', 'type', 'choices', 'context', 'answer'
    ])

    print("--- 整形後データ例 ---")
    print("prompt:", dataset[0]["prompt"])
    print("label:", dataset[0]["label"])

    # --- 3.5 トークナイズ ---
    def tokenize_function(examples):
        # prompt部分（固定長512）
        model_inputs = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        # label部分（固定長16）
        label_inputs = tokenizer(
            text_target=examples["label"],
            truncation=True,
            max_length=16,
            padding="max_length",
            add_special_tokens=False
        )

        # 結合
        input_ids = model_inputs["input_ids"] + label_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"] + label_inputs["attention_mask"]

        # prompt部分は -100 にする
        labels = [-100] * len(model_inputs["input_ids"]) + label_inputs["input_ids"]

        # 確認用
        assert len(input_ids) == len(attention_mask) == len(labels), "長さ不一致があります"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["prompt", "label"])

    # --- 4. トレーニング ---
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=20,
        learning_rate=2e-5,
        fp16=True,
        save_strategy="epoch"
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
    )

    print("トレーニングを開始します...")
    trainer.train()
    print("トレーニングが完了しました。")

    adapter_path = f"{args.output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    print(f"学習済みLoRAアダプタを {adapter_path} に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning Script")
    parser.add_argument("--model_id", type=str, default="EleutherAI/gpt-neo-125M", help="ベースとなるモデルID")
    parser.add_argument("--dataset_name", type=str, default="bigbio/med_qa", help="使用するデータセット名")
    parser.add_argument("--output_dir", type=str, default="./qlora-finetune-output", help="結果の出力先ディレクトリ")
    
    args = parser.parse_args()
    main(args)
