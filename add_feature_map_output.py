"""
https://blog.roboflow.com/yolo-object-detection/  # v8
https://github.com/ultralytics/ultralytics/issues/22566  # v11

Script to add YOLO12n P4 feature map output (40x40x128) to the ONNX model.
This modifies best_vis.onnx to create best_vis_with_feature_map.onnx
with an additional output from the neck layer: /model/model.21/Concat_5_output_0

Output format:
- outputs[0]: detection output [1, 300, 6]
- outputs[1]: feature map [1, 5, 8400]
"""

import onnx
from onnx import helper


def add_feature_map_output(input_model_path, output_model_path, feature_layer_name, shape=None):
    """Add a feature map output to an ONNX model.

    Args:
        input_model_path: Path to the input ONNX model (e.g., 'models/best_vis.onnx')
        output_model_path: Path to save the modified model (e.g., 'models/best_vis_with_feature_map.onnx')
        feature_layer_name: Name of the layer to add as output (e.g., '/model/model.21/Concat_5_output_0')
    """
    # Load the original ONNX model
    model = onnx.load(input_model_path)

    # Get the existing outputs
    existing_outputs = [output.name for output in model.graph.output]
    print(f"Existing outputs: {existing_outputs}")

    # Check if the feature layer exists in the model
    all_tensors = set()
    for node in model.graph.node:
        all_tensors.update(node.output)

    if feature_layer_name not in all_tensors:
        print(f"Warning: Layer '{feature_layer_name}' not found in model")
        print("Available intermediate tensors (sample):")
        for tensor in list(all_tensors)[:20]:
            print(f"  - {tensor}")
        raise ValueError(f"Layer '{feature_layer_name}' not found in the model")

    # Add the feature map as an additional output
    # We need to create a ValueInfoProto for the new output
    if not shape:
        shape = [1, 5, 8400]  # Shape: [batch, channels, height, width]
    feature_output = helper.make_tensor_value_info(
        feature_layer_name,
        onnx.TensorProto.FLOAT,
        shape
    )

    # Add the new output to the model
    model.graph.output.append(feature_output)

    # Save the modified model
    onnx.save(model, output_model_path)

    print(f"\nModel saved successfully to: {output_model_path}")
    print("New outputs:")
    for i, output in enumerate(model.graph.output):
        print(f"  outputs[{i}]: {output.name}")


if __name__ == "__main__":
    # INPUT_MODEL = "models/best_vis.onnx"
    # INPUT_MODEL = "models/best_vis_with_8400.onnx"
    # OUTPUT_MODEL = "models/best_vis_with_8400_test.onnx"

    INPUT_MODEL = "models/yolo12n.onnx"
    OUTPUT_MODEL = "models/yolo12n_with_8400_test.onnx"


    # FEATURE_LAYER = "model.21/Concat_5_output_0"
    FEATURE_LAYER = "/model.14/cv2/act/Mul_output_0"  #
    shape = [1, 64, 80, 80]
    print(f"Converting {INPUT_MODEL} to {OUTPUT_MODEL}")
    print(f"Adding feature map output from layer: {FEATURE_LAYER}")

    add_feature_map_output(INPUT_MODEL, OUTPUT_MODEL, FEATURE_LAYER, shape=shape)

    # print("\nDone! You can now use the model with:")
    # print("  outputs = session.run(None, {'images': input_img})")
    # print("  detections = outputs[0]  # [1, 300, 6]")
    # print("  feature_map = outputs[1]  # [1, 5, 8400]")


# ========= ========= ========= ========= ========= =========


    # # onnx output Head feature (models/best_vis_with_8400_3.onnx)
    # # onnx output Head feature (models/best_vis_with_8400_3.onnx)
    from_save = [
        "models/best_vis_with_8400.onnx",  # original file
        "models/best_vis_with_8400_1.onnx",  # middle file
        "models/best_vis_with_8400_2.onnx",  # middle file
        "models/best_vis_with_8400_3.onnx"  # final file
    ]
    # None, 14, 17, 20
    add_FEATURE = [
        ("/model/model.14/cv2/act/Mul_output_0", [1, 64, 80, 80]),
        ("/model/model.17/cv2/act/Mul_output_0", [1, 128, 40, 40]),
        ("/model/model.20/cv2/act/Mul_output_0", [1, 256, 20, 20])
    ]


    # /model.14/cv2/act/Mul_output_0  # Conv * Sigmoid
    # /model.14/cv2/conv/Conv_output_0 # (Conv)
    # /model.14/cv2/act/Sigmoid_output_0  (Sigmoid)
    # from_save = [
    #     "models/best_vis_with_8400.onnx",  # original file
    #     "models/best_vis_with_sig_8400_1.onnx",  # middle file
    #     "models/best_vis_with_sig_8400_2.onnx",  # middle file
    #     "models/best_vis_with_sig_8400_3.onnx"  # final file
    # ]
    # # Take Sigmoid output
    # add_FEATURE = [
    #     ("/model/model.14/cv2/act/Sigmoid_output_0", [1, 64, 80, 80]),
    #     ("/model/model.17/cv2/act/Sigmoid_output_0", [1, 128, 40, 40]),
    #     ("/model/model.20/cv2/act/Sigmoid_output_0", [1, 256, 20, 20])
    # ]
    # # onnx output Head feature (models/best_vis_with_8400_3.onnx)
    # # onnx output Head feature (models/best_vis_with_8400_3.onnx)


    # === [Custom yolov12n model] backbone P3, P4, P5 ===
    # === [Custom yolov12n model] backbone P3, P4, P5 ===
    # from_save = [  # backbone P3, P4, P5
    #     "models/best_vis_with_8400.onnx",  # original file
    #     "models/best_vis_with_8400_b1.onnx",  # middle file
    #     "models/best_vis_with_8400_b2.onnx",  # middle file
    #     "models/best_vis_with_8400_b3.onnx"  # final file
    # ]

    # add_FEATURE = [ # 4, 6, 8
    #     ("/model/model.4/cv1/act/Mul_output_0", [1, 64, 80, 80]),
    #     ("/model/model.6/cv1/act/Mul_output_0", [1, 64, 40, 40]),
    #     ("/model/model.8/cv1/act/Mul_output_0", [1, 128, 20, 20])
    # ]
    # === [Custom yolov12n model] backbone P3, P4, P5 ===
    # === [Custom yolov12n model] backbone P3, P4, P5 ===



    # # onnx output Head feature (models/yolo model test.onnx)
    # from_save = [
    #     "models/yolo12n.onnx",  # original file
    #     "models/yolo12n_with_8400_1.onnx",  # middle file
    #     "models/yolo12n_with_8400_2.onnx",  # middle file
    #     "models/yolo12n_with_8400_3.onnx",  # final file
    # ]

    # None, 14, 17, 20
    # add_FEATURE = [
    #     ("/model.14/cv2/act/Mul_output_0", [1, 64, 80, 80]),
    #     ("/model.17/cv2/act/Mul_output_0", [1, 128, 40, 40]),
    #     ("/model.20/cv2/act/Mul_output_0", [1, 256, 20, 20])
    # ]

    # None, 14, 17, 20 (Conv, before activation sigmoid)
    # add_FEATURE = [
    #     ("/model.14/cv2/conv/Conv_output_0", [1, 64, 80, 80]),
    #     ("/model.17/cv2/conv/Conv_output_0", [1, 128, 40, 40]),
    #     ("/model.20/cv2/conv/Conv_output_0", [1, 256, 20, 20])
    # ]

    # # None, [BackBone, P3, P4, P5] 4, 6, 8 (Conv, before activation sigmoid)
    # # https://arxiv.org/pdf/2408.15857v1
    # # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12.yaml
    # add_FEATURE = [
    #     ("/model.4/cv1/act/Mul_output_0", [1, 64, 80, 80]),
    #     ("/model.6/cv1/act/Mul_output_0", [1, 64, 40, 40]),
    #     ("/model.8/cv1/act/Mul_output_0", [1, 128, 20, 20])
    # ]


    # add_FEATURE = [
    #     ("/model.1/conv/Conv_output_0", [1, 16, 320, 320]),  # P2 超前面 (受光線影響比較明顯)
    #     ("/model.3/act/Mul_output_0", [1, 64, 80, 80]),
    #     ("/model.8/cv1/act/Mul_output_0", [1, 128, 20, 20])
    # ]
    # # name: /model.2/cv1/act/Mul_output_0


    for idx, _FEATURE in enumerate(add_FEATURE):
        IN_MODEL = from_save[idx]
        OUT_MODEL = from_save[idx + 1]
        _FEATURE_LAYER_NAME = _FEATURE[0]
        _FEATURE_LAYER_SHAPE = _FEATURE[1]
        print(f"Converting {IN_MODEL} to {OUT_MODEL}")
        print(f"Adding feature map output from layer: {_FEATURE_LAYER_NAME}")
        add_feature_map_output(
            IN_MODEL, OUT_MODEL,
            _FEATURE_LAYER_NAME, shape=_FEATURE_LAYER_SHAPE
        )