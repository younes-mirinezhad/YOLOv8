"""
YOLO Training Main Script
Entry point for YOLO model training with user inputs
"""

from pathlib import Path
import argparse
from Trainer import YOLOTrainer
from Exporter import YOLOExporter

# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "BaseFiles/config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO Training Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode. When set, --config must be provided.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to training config YAML (required when --train is set)",
    )
    parser.add_argument(
        "--export",
        dest="export_type",
        choices=("onnx", "onnx_e2e"),
        default=None,
        help="Export format (e.g., onnx, onnx_e2e)",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=Path,
        default=None,
        help="Path to trained model file (required when --export is set without --train)",
    )

    args = parser.parse_args()

    # Validation: if --train is set, --config must be provided
    if args.train and not args.config_path:
        parser.error("--config is required when --train is set")

    # Validation: if --export is set without --train, --model must be provided
    if args.export_type and not args.train and not args.model_path:
        parser.error("--model is required when --export is set without --train")

    # Validation: at least one action must be specified
    if not args.train and not args.export_type:
        parser.error("At least one action must be specified: --train or --export")

    return args


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    print("=" * 60)
    print("YOLO Pipeline")
    print("=" * 60)

    # Display configuration
    if args.train:
        print(f"Mode: Training")
        config_path = args.config_path if args.config_path else DEFAULT_CONFIG_PATH
        print(f"Configuration file: {config_path}")

    if args.export_type:
        print(f"Export format: {args.export_type}")
    if args.model_path:
        print(f"Model path: {args.model_path}")

    print("=" * 60)

    trainer = None

    try:
        # Training phase
        if args.train:
            config_path = args.config_path if args.config_path else DEFAULT_CONFIG_PATH

            # Initialize the trainer with the config path
            print("\nInitializing YOLO Trainer...")
            trainer = YOLOTrainer(config_path=config_path)

            # Start training
            print("\nStarting training process...\n")
            results = trainer.train()

            print("\n" + "=" * 60)
            print("Training pipeline completed successfully!")
            print("=" * 60)

        # Export phase
        if args.export_type:
            print("\nInitializing YOLO Exporter...")

            if trainer and trainer.get_model():
                # Export the trained model
                exporter = YOLOExporter(model=trainer.get_model())
            elif args.model_path:
                # Export from existing model file
                exporter = YOLOExporter(model_path=args.model_path)
            else:
                # This should not happen due to argument validation
                print("\nError: No model available for export.")
                return 1

            print("\nStarting exporting process...\n")
            exporter.export(export_type=args.export_type)

            print("\n" + "=" * 60)
            print("Exporting model completed successfully!")
            print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check your configuration file path.")
        return 1
    except Exception as e:
        print(f"\nError during execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
