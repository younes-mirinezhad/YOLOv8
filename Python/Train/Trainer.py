"""
YOLO Model Trainer
Supports training YOLOv8, YOLOv10, YOLOv11, YOLOv26
"""

from pathlib import Path
import argparse
from ultralytics import YOLO
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "BaseFiles/config.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO training entrypoint")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config YAML",
    )
    return parser.parse_args()


def resolve_paths(config: dict, base_dir: Path) -> dict:
    resolved = dict(config)
    for key in ("yaml_path", "hyperparameters_path", "project"):
        raw_value = resolved.get(key)
        if not raw_value:
            continue
        path_value = Path(raw_value)
        if not path_value.is_absolute():
            path_value = (base_dir / path_value).resolve()
        resolved[key] = str(path_value)
    return resolved


def train_yolo(config: dict, config_dir: Path):
    config = resolve_paths(config, config_dir)
    # Load training configs
    model_name = config["model_name"]
    yaml_path = config["yaml_path"]
    project = config["project"]
    name = config["name"]
    epochs = config["epochs"]
    imgsz = config["imgsz"]
    batch = config["batch"]
    device = config["device"]
    lr0 = config["lr0"]
    lrf = config["lrf"]
    optimizer = config["optimizer"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]
    warmup_epochs = config["warmup_epochs"]
    warmup_momentum = config["warmup_momentum"]
    warmup_bias_lr = config["warmup_bias_lr"]
    patience = config["patience"]
    save = config["save"]
    save_period = config["save_period"]
    cache = config["cache"]
    workers = config["workers"]
    exist_ok = config["exist_ok"]
    pretrained = config["pretrained"]
    verbose = config["verbose"]
    val = config["val"]
    plots = config["plots"]
    hyperparameters_path = config["hyperparameters_path"]

    if not Path(hyperparameters_path).exists():
        raise FileNotFoundError(
            f"Hyperparameters file not found: {hyperparameters_path}"
        )

    hyperparams = load_yaml(Path(hyperparameters_path))

    # Load the model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Validate YAML path
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    print("-" * 50)
    print(f"Dataset YAML: {yaml_path}")
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Learning rate: {lr0} -> {lr0 * lrf}")
    print(f"  Optimizer: {optimizer}")
    print("-" * 50)

    # Start training
    print("\nStart training!")
    results = model.train(
        data=yaml_path,
        project=project,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        patience=patience,
        save=save,
        save_period=save_period,
        cache=cache,
        workers=workers,
        exist_ok=exist_ok,
        pretrained=pretrained,
        verbose=verbose,
        val=val,
        plots=plots,
        **hyperparams,
    )

    print("\nTraining completed!")
    print("Training results saved.")

    return results


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_yaml(config_path)
    train_yolo(config, config_path.parent)
