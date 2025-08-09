import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. モデルとアダプタのロード（よりシンプルな方法） ---
model_id = "EleutherAI/gpt-neo-125M"
adapter_path = "./qlora-finetune-output/final_adapter"

print("ベースモデルをロード中 (bfloat16)...")
# QLoRAの4bit量子化を使わずに、通常の16bit精度でモデルを直接GPUにロードします
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 精度を指定
    device_map="auto"
)

print("LoRAアダプタをマージ中...")
# ベースモデルに学習済みLoRAアダプタをマージ
model = PeftModel.from_pretrained(base_model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. シンプルなプロンプトで推論 ---
# 学習時と同じ形式のプロンプト
prompt = "Question: Which part of the brain is primarily responsible for vision?\n\nOptions:\nA: Frontal lobe\nB: Occipital lobe\nC: Temporal lobe\nD: Parietal lobe\n\nAnswer:"

# さらにシンプルな、どんなモデルでも答えられそうなプロンプト
# prompt = "The capital of Japan is" 

print(f"プロンプト: '{prompt}'")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("生成を実行中...")
# 最もシンプルな設定で生成
outputs = model.generate(**inputs, max_new_tokens=20)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- 結果 ---")
print(decoded_output)
print("------------")