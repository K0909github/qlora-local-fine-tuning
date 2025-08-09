from peft import PeftModel
from transformers import AutoModelForCausalLM

# --- 1. 設定項目 ---

# ベースモデルのID
model_id = "EleutherAI/gpt-neo-125M"

# 保存したLoRAアダプタのローカルパス
adapter_path = "./qlora-finetune-output/final_adapter"


hub_repo_id = "K0909/brain-qa-adapter"


# --- 2. アップロード処理 ---

print("ベースモデルをロード中...")
# アップロードのためだけなので、CPU上でロードすればOKです
base_model = AutoModelForCausalLM.from_pretrained(model_id)

print("LoRAアダプタをロード中...")
# ベースモデルに学習済みLoRAアダプタをマージ
model = PeftModel.from_pretrained(base_model, adapter_path)

print(f"'{hub_repo_id}' にアップロードしています...")
# model.push_to_hub() を呼び出すだけでアップロードが完了します
model.push_to_hub(hub_repo_id)

print("✅ アップロードが完了しました！")