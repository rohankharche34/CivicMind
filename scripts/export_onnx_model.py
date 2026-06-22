import os
import sys
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "onnx-model",
)
os.makedirs(output_dir, exist_ok=True)

print("Exporting all-MiniLM-L6-v2 to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"ONNX model exported to {output_dir}")
