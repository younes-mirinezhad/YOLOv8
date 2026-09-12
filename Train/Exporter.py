"""
YOLO Model Exporter
Handles exporting trained YOLO models to various formats
"""

from pathlib import Path
from ultralytics import YOLO


class YOLOExporter:
    """
    YOLO Exporter class for exporting trained models to different formats.
    """

    SUPPORTED_FORMATS = [
        "onnx",
        "onnx_e2e",
        "torchscript",
        "coreml",
        "tensorflow",
        "tflite",
        "pb",
        "engine",
    ]

    def __init__(self, model=None, model_path: Path = None):
        """
        Initialize the YOLO Exporter.

        Args:
            model: An already loaded YOLO model instance (from trainer).
            model_path: Path to a saved model file. Used if model is not provided.
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"Loading model from: {model_path}")
            self.model = YOLO(str(model_path))
        else:
            raise ValueError("Either 'model' or 'model_path' must be provided.")

    def export(self, export_type: str = "onnx", **kwargs):
        """
        Export the trained model to specified format.

        Args:
            export_type: Export format (e.g., 'onnx', 'onnx_e2e', 'torchscript', 'coreml', etc.)
            **kwargs: Additional arguments to pass to the export function.

        Returns:
            Export path or result from the ultralytics export function.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot export.")

        # Validate export format
        if export_type not in self.SUPPORTED_FORMATS:
            print(f"Warning: '{export_type}' may not be officially supported.")
            print(f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")

        print(f"\nExporting model to {export_type}...")
        print("-" * 50)

        try:
            if export_type == "onnx":
                result = self.model.export(
                    format="onnx",
                    simplify=True,   # Remove redundant nodes/reshapes/transposes (onnxsim)
                    dynamic=True,    # Fixed input shape (1×3×640×640); set True for Nx3xHxW flexibility
                    half=False,      # FP32 weights; set True for FP16 (GPU/TensorRT only)
                    nms=False,       # Raw predictions output; set True to embed NMS in the graph
                    opset=17,        # ONNX opset version (17 = latest stable for most runtimes)
                    **kwargs,
                )
            elif export_type == "onnx_e2e":
                result = self.model.export(
                    format="onnx",
                    simplify=True,   # Remove redundant nodes/reshapes/transposes (onnxsim)
                    dynamic=False,   # Fixed input shape (1×3×640×640); set True for Nx3xHxW flexibility
                    half=False,      # FP32 weights; set True for FP16 (GPU/TensorRT only)
                    nms=True,        # Embed NMS in the graph for end-to-end inference
                    opset=17,        # ONNX opset version (17 = latest stable for most runtimes)
                    **kwargs,
                )
            else:
                result = self.model.export(format=export_type, **kwargs)

            print(f"Export to {export_type} completed!")
            print("-" * 50)
            return result

        except Exception as e:
            print(f"Error during export: {e}")
            raise

    def get_model(self):
        """Get the loaded model instance."""
        return self.model
