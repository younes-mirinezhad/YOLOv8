import os
import json
import shutil
import cv2

# Update these to match your dataset locations
INPUT_DIR = (
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/3_Approved/V2_DarkSmoke"
)
OUTPUT_DIR = "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format"

# Extend/adjust as needed. Unknown labels are skipped.
LABEL_MAP = {"Flame": 0, "Smoke": 1, "Flare": 2, "Snow": 3}


def get_subfolders(directory: str):
    """Return subfolder names in a directory."""

    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]


def convert_labelme_folder(input_dir: str, output_dir: str) -> int:
    counter = 0
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, file_name)
        txt_path = os.path.join(output_dir, file_name.replace(".json", ".txt"))
        convert_labelme_to_yolo_det(json_path, txt_path)

        img_src = json_path.replace(".json", ".jpg")
        img_dst = txt_path.replace(".txt", ".jpg")
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)
        counter += 1
    return counter


def convert_labelme_to_yolo_det(json_path: str, output_txt_path: str) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    try:
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
    except KeyError:
        img_path = json_path.replace(".json", ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found for {json_path}")
        img_height, img_width, _ = img.shape

    with open(output_txt_path, "w") as f_out:
        for shape in data.get("shapes", []):
            label = shape.get("label")
            cls_id = LABEL_MAP.get(label, -1)
            if cls_id == -1:
                continue

            shape_type = shape.get("shape_type")
            points = shape.get("points", [])
            if shape_type == "rectangle" and len(points) == 2:
                (x1, y1), (x2, y2) = points
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
            elif points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            else:
                continue

            x_center = ((x_min + x_max) / 2.0) / img_width
            y_center = ((y_min + y_max) / 2.0) / img_height
            box_w = (x_max - x_min) / img_width
            box_h = (y_max - y_min) / img_height

            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
            f_out.write(line)


if __name__ == "__main__":
    folders = get_subfolders(INPUT_DIR)

    for folder in folders:
        input_path = os.path.join(INPUT_DIR, folder)
        output_path = os.path.join(OUTPUT_DIR, folder)

        if not os.path.exists(input_path):
            print(f"*** Wrong input dir: {input_path}")
            continue

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        count = convert_labelme_folder(input_path, output_path)
        print(f'"{folder}",  # {count}')
