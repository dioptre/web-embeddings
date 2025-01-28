# convert_to_onnx.py
import torch
from transformers import AutoModel, AutoTokenizer
import onnx

def convert_gte_small_to_onnx(
    model_name="thenlper/gte-small",
    output_path="model.onnx",
    opset_version=13,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Example input for tracing
    text = "Hello from GTE-Small model!"
    tokens = tokenizer(text, return_tensors="pt")

    # Torch -> ONNX
    torch.onnx.export(
        model,
        (tokens["input_ids"], tokens["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],  # or 'sentence_embedding', depends on the model
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,
        opset_version=opset_version,
    )
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    convert_gte_small_to_onnx()
