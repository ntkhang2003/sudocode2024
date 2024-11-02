import gradio as gr
import requests

# Hàm gọi API FastAPI
def classify_text(text):
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": text})
    if response.status_code == 200:
        return response.json()["label"]
    else:
        return "Error"

# Khởi tạo giao diện Gradio
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="text",
    title="PhoBERT Sentiment Analysis",
    description="Nhập văn bản để phân loại cảm xúc (Positive/Negative)"
)

# Chạy Gradio
iface.launch()
