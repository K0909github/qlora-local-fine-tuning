# QLoRA Local Fine-Tuning

このリポジトリは、QLoRA（Quantized Low-Rank Adapter）を用いて、EleutherAI/gpt-neo-125Mモデルを医学QAデータセット（bigbio/med_qa）でローカル微調整するサンプルです。

Hugging Face公開済みモデル: [K0909/brain-qa-adapter](https://huggingface.co/K0909/brain-qa-adapter)

## 構成
- `train.py` : QLoRAによる微調整スクリプト
- `inference.py` : 学習済みモデルで推論を行うスクリプト
- `upload.py` : Hugging Face Hubへのアップロードスクリプト
- `requirements.txt` : 必要なPythonパッケージ
- `qlora-finetune-output/` : 学習結果（各チェックポイント、最終アダプタ）

## 使い方

### 1. 学習
```bash
python train.py
```

### 2. 推論
```bash
python inference.py
```

#### 推論出力例
```
モデルに質問しています...
--------------------
Question: Which part of the brain is primarily responsible for vision?

Options:
A: Frontal lobe
B: Occipital lobe
C: Temporal lobe
D: Parietal lobe

Answer: Occipital lobe
```

### 3. Hugging Face Hubへのアップロード
`upload.py` を実行すると、最終アダプタ（`qlora-finetune-output/final_adapter`）がHugging Face Hubにアップロードされます。

```bash
python upload.py
```

#### アップロード例
```
Uploading adapter to Hugging Face Hub...
Successfully uploaded to https://huggingface.co/<ユーザー名>/<リポジトリ名>
```

## Hugging Faceでのロード方法
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "EleutherAI/gpt-neo-125M"
adapter_path = "K0909/brain-qa-adapter"

base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Question: Which part of the brain is primarily responsible for vision?\n\nOptions:\nA: Frontal lobe\nB: Occipital lobe\nC: Temporal lobe\nD: Parietal lobe\n\nAnswer: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## ライセンス
MIT

## 作者
GitHub: K0909github
