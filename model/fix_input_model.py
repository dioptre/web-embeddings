# fix_input_dtype.py
import onnx

def convert_input_to_int32(onnx_path_in, onnx_path_out):
    model = onnx.load(onnx_path_in)
    for input_tensor in model.graph.input:
        # If it's INT64, change to INT32
        if input_tensor.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT32
    onnx.save(model, onnx_path_out)
    print(f"Converted {onnx_path_in} -> {onnx_path_out} with int32 inputs.")

if __name__ == "__main__":
    convert_input_to_int32("model/model_O4.onnx", "model/model_O4_int32.onnx")
