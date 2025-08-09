import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# --- モデルとトークナイザ ---
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

# --- 推論 ---
question = "Which part of the brain is primarily responsible for vision?"
options = "A: Frontal lobe\nB: Occipital lobe\nC: Temporal lobe\nD: Parietal lobe"
prompt = f"Question: {question}\n\nOptions:\n{options}\n\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
    max_new_tokens=10,  # 答えだけ生成
    temperature=0.0,    # Greedy生成
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

print("モデルに質問しています...")
print("--------------------")
print(prompt, end=" ")

outputs = model.generate(**inputs, generation_config=generation_config)

output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(output_text.strip())
