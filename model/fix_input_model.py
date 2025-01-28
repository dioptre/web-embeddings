# fix_input_dtype.py
import onnx

def convert_input_to_int32(onnx_path_in, onnx_path_out):
    model = onnx.load(onnx_path_in)
    
    # Print initial input types
    print("Initial input tensor types:")
    for input_tensor in model.graph.input:
        print(f"- {input_tensor.name}: {input_tensor.type.tensor_type.elem_type}")
        
    # Convert inputs
    changes_made = False
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            print(f"Converting {input_tensor.name} from INT64 to INT32")
            input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT32
            changes_made = True
    
    # Print final input types
    print("\nFinal input tensor types:")
    for input_tensor in model.graph.input:
        print(f"- {input_tensor.name}: {input_tensor.type.tensor_type.elem_type}")
    
    if not changes_made:
        print("\nWarning: No changes were made to the model!")
    else:
        onnx.save(model, onnx_path_out)
        print(f"\nConverted {onnx_path_in} -> {onnx_path_out} with int32 inputs.")

if __name__ == "__main__":
    # Update the path to match your model file
    convert_input_to_int32("model/model.onnx", "model/model_int32.onnx")
