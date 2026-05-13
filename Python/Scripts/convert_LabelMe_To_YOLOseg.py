import os, json, shutil, cv2

input_dir = "/media/HDD_2/Dataset/Watchtower/3_Approved/V2_DarkSmoke"
output_dir = "/media/HDD_2/Dataset/Watchtower/4_YOLO_Format"

LABEL_MAP = {"Flame": 0, "Smoke": 1, "Flare": 2}


def get_subfolders(directory):
    """
    Returns a list of all subfolder names within the given directory.
    """
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]


def convert_labelme_to_yolo(input_dir, output_dir):
    counter = 0
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, file_name)
        txt_path = os.path.join(output_dir, file_name.replace(".json", ".txt"))
        convert_labelme_to_yolo_seg(json_path, txt_path)
        shutil.copy(
            json_path.replace(".json", ".jpg"), txt_path.replace(".txt", ".jpg")
        )
        counter += 1
    return counter


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
    folders = get_subfolders(input_dir)

    for dir in folders:
        input = os.path.join(input_dir, dir)
        output = os.path.join(output_dir, dir)

        if os.path.exists(input):
            if not os.path.exists(output):
                os.mkdir(output)

            count = convert_labelme_to_yolo(input, output)
            print(f'"{dir}",  # {count}')
        else:
            print(f"*** Wrong input dir: {input}")
