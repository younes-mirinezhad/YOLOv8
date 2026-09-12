import onnx
from onnx.onnx_pb import StringStringEntryProto


def write(input_model_path, metadata, output_model_path):
    model = onnx.load(input_model_path)

    # Add metadata to model
    for k, v in metadata.items():
        entry = StringStringEntryProto(key=k, value=v)
        model.metadata_props.append(entry)

    # Save updated model
    onnx.save(model, output_model_path)
    print("✅ Metadata added and saved to yolov8_with_metadata.onnx")


def read(model_path):
    # Load model
    model = onnx.load(model_path)

    # Read metadata
    print("📌 Model Metadata:")
    for prop in model.metadata_props:
        print(f"{prop.key}: {prop.value}")


if __name__ == "__main__":
    input_model_path = "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Development/Models/Segmentation/V3_AryaSasolـbest_b1_640.onnx"
    output_model_path = "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Development/Models/Segmentation/yolov8_with_metadata.onnx"
    metadata = {
        "train_date": "2025-08-23",
        "tag": "V3_AryaSasol",
        "train_dataset": "D02_20250103150721_01, D02_20250103160132_01, D02_20250103175716_smoke_night_01",
        "accuracy": "Mask mAP50-95: [all: 0.831, Flame: 0.944, Smoke: 0.718]",
    }
    write(input_model_path, metadata, output_model_path)
    read(output_model_path)
