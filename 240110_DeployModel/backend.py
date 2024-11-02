from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load mô hình và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("phobert_sa")
tokenizer = AutoTokenizer.from_pretrained("phobert_sa")

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Định nghĩa yêu cầu đầu vào
class TextInput(BaseModel):
    text: str

# Mapping nhãn
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Tạo route cho phân loại
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    # Tokenize và chuyển đổi input thành tensor
    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
    # Ánh xạ nhãn
    label = label_map.get(prediction, "unknown")
    return {"label": label, "score": logits.softmax(dim=-1).tolist()}
