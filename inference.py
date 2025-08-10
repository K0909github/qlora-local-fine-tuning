import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "EleutherAI/gpt-neo-125M"
adapter_path = "./qlora-finetune-output/final_adapter"

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# モデル読み込み
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRAアダプタ適用
model = PeftModel.from_pretrained(base_model, adapter_path)

# トークナイザ
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 質問
question = "Which part of the brain is primarily responsible for vision?"
options = "A: Frontal lobe\nB: Occipital lobe\nC: Temporal lobe\nD: Parietal lobe"
prompt = f"Question: {question}\n\nOptions:\n{options}\n\nAnswer:"

# トークン化
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("モデルに質問しています...")
print("--------------------")
print(prompt, end=" ")

# 直接generateにパラメータを渡す
outputs = model.generate(
    **inputs,
    max_new_tokens=5,         # 答えだけ出す
    temperature=0.0,          # Greedy生成
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# デコード＆余分な部分を削除
output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
answer = output_text.strip().split("\n")[0]  # 最初の行だけ
print(answer)
