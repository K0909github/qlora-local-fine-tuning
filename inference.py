import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# --- 1. モデルとトークナイザの準備 ---
model_id = "EleutherAI/gpt-neo-125M"
adapter_path = "./qlora-finetune-output/final_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. 推論の実行 ---

question = "Which part of the brain is primarily responsible for vision?"
options = "A: Frontal lobe\nB: Occipital lobe\nC: Temporal lobe\nD: Parietal lobe"
prompt = f"Question: {question}\n\nOptions:\n{options}\n\nAnswer: "

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


# 生成に関する設定を一つのオブジェクトにまとめる
generation_config = GenerationConfig(
    max_new_tokens=200,
    temperature=1.0,
    repetition_penalty=1.2,
    do_sample=True,  # サンプリングを有効にしてtemperatureを使えるようにする
    pad_token_id=tokenizer.eos_token_id # pad_token_idを明示的に設定
)

print("モデルに質問しています...")
print("--------------------")
print(prompt, end="")

# model.generateに設定オブジェクトを渡す
outputs = model.generate(**inputs, generation_config=generation_config)

# 生成されたトークンをデコードして表示
output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
# "Answer:"以降のみ抽出
if "Answer:" in output_text:
    answer = output_text.split("Answer:", 1)[1].strip()
else:
    answer = output_text.strip()
print(answer)