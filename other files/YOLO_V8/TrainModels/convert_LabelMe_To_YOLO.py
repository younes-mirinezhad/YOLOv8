import os, json, shutil, cv2

images_path = "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/3_Approved/LabelME/D02_20250103150721_02_Arshia"
output_path = "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/D02_20250103150721_02_Arshia"

LABEL_MAP = {"Flame": 0, "Smoke": 1}


def convert_labelme_to_yolo(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".json"):
            continue

        json_path = os.path.join(images_path, file_name)
        txt_path = os.path.join(output_path, file_name.replace(".json", ".txt"))
        convert_labelme_to_yolo_seg(json_path, txt_path)
        shutil.copy(
            json_path.replace(".json", ".jpg"), txt_path.replace(".txt", ".jpg")
        )


def convert_labelme_to_yolo_seg(json_path, output_txt_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    try:
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
    except:
        img_path = json_path.replace(".json", ".jpg")
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

    with open(output_txt_path, "w") as f_out:
        for shape in data["shapes"]:
            label = shape["label"]
            cls_id = LABEL_MAP.get(label, -1)
            if cls_id == -1 or shape["shape_type"] != "polygon":
                continue

            points = shape["points"]
            # Normalize and flatten the points
            norm_points = []
            for x, y in points:
                x_norm = x / img_width
                y_norm = y / img_height
                norm_points.extend([f"{x_norm:.10f}", f"{y_norm:.10f}"])

            # Write to output file
            line = f"{cls_id} " + " ".join(norm_points) + "\n"
            f_out.write(line)


if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    convert_labelme_to_yolo(images_path, output_path)
    print("----- Done -----")
