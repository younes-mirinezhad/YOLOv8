import os
import shutil
import random

input_folders = [
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/10_01",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/10_02",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/10_03",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/10_06",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/11_01",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/D02_20250103150721_01_Arshia",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/D02_20250103150721_02_Arshia",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/D02_20250103160132_01_Ali",
    "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/4_YOLO_Format/D02_20250103175716_smoke_night_01",
]
output_dir = "TrainModels/sampleData/"
test_ratio = 0.1
allowed_extensions = {".jpg", ".jpeg"}


def make_dirs(output_dir):
    for sub in ["train", "val"]:
        p_image = os.path.join(output_dir, "images", sub)
        if os.path.exists(p_image):
            shutil.rmtree(p_image)
        os.makedirs(os.path.join(output_dir, "images", sub), exist_ok=True)

        p_label = os.path.join(output_dir, "labels", sub)
        if os.path.exists(p_label):
            shutil.rmtree(p_label)
        os.makedirs(os.path.join(output_dir, "labels", sub), exist_ok=True)


def main():
    make_dirs(output_dir)

    for dataset in input_folders:
        files = os.listdir(dataset)
        imgs = [
            f for f in files if os.path.splitext(f)[1].lower() in allowed_extensions
        ]

        n_test = max(1, int(len(imgs) * test_ratio)) if imgs else 0
        test_files = set(random.sample(imgs, n_test)) if imgs and n_test > 0 else set()

        for img in imgs:
            base, ext = os.path.splitext(img)
            txt = base + ".txt"
            src_img = os.path.join(dataset, img)
            src_txt = os.path.join(dataset, txt)

            if img in test_files:
                dest_img = os.path.join(output_dir, "images", "val", img)
                dest_txt = os.path.join(output_dir, "labels", "val", txt)
            else:
                dest_img = os.path.join(output_dir, "images", "train", img)
                dest_txt = os.path.join(output_dir, "labels", "train", txt)

            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_txt, dest_txt)


if __name__ == "__main__":
    main()
