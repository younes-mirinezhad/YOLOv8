import os
import shutil
import random

baseDIR = "/media/HDD_2/Dataset/Watchtower/4_YOLO_Format"
# Days ignored: 20250801_1816_SF, 20250917_0716, 20250823_1556_SF, 20250801_1826_SF
# Night ignored:
folders = [
    "20250103_0717_P01",  # 50
    "20251209_1615_Cloud",  # 73
    "20251209_0900_Cloud",  # 73
    "20260214_1200_Flame_Smoke",  # 59
    "Snapshots_14070711_14040909",  # 50
    "20250103_15_P01",  # 50
    "20250903_1406",  # 74
    "20250720_09",  # 75
    "20250809_12",  # 76
    "20251209_1620_Cloud",  # 73
    "20260214_1000_Flame_Smoke",  # 56
    "20250926_1756_Flame",  # 73
    "20250801_1826_SF",  # 76
    "20251119_0625_Flame_Cloud",  # 73
    "20250103_16_P01",  # 50
    "20250103_16_P02",  # 50
    "20260214_0755_Flame_Smoke",  # 56
    "20250801_1816_SF",  # 76
    "20251213_0840_SmallFlame_Cloud",  # 72
    "20250823_1556_SF",  # 17
    "20250928_0856",  # 73
    "20250103_08_SF_BG_P01",  # 50
    "20250103_15_P02",  # 50
    "20260303_1510_SmallFlame_Cloud",  # 59
    "20250825_12",  # 75
    "20250103_1703_P04",  # 50
    "20251208_1700_Cloud",  # 73
    "20250917_15",  # 74
    "20260214_1400_Flame",  # 57
    "20250921_0816",  # 74
    "20260301_0627_Cloud",  # 28
    "20251209_0715_Cloud",  # 73
    "20260301_0640_Cloud",  # 59
    "20250930_0736",  # 73
    "20251209_1520_Cloud",  # 73
    "20251003_1151_flame_LightSmoke",  # 72
    "20250926_1026",  # 73
    "20250903_1411",  # 75
    "20260214_0655_Flame_Smoke",  # 57
    "20260214_0900_Flame_Smoke",  # 56
    "20250927_1616",  # 72
    "20251022_1452_Cloud",  # 73
    "20260214_1100_Flame_Smoke",  # 56
    "20250917_0716",  # 75
    "20251119_0905_Flame_Cloud",  # 73
    "20251123_1315_Cloud",  # 73
    "20260214_1300_Flame",  # 58
]

output_dir = "Watchtower/dataset"
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

    for dir in folders:
        dirPath = os.path.join(baseDIR, dir)
        files = os.listdir(dirPath)
        imgs = [
            f for f in files if os.path.splitext(f)[1].lower() in allowed_extensions
        ]

        n_test = max(1, int(len(imgs) * test_ratio)) if imgs else 0
        test_files = set(random.sample(imgs, n_test)) if imgs and n_test > 0 else set()

        for img in imgs:
            base, ext = os.path.splitext(img)
            txt = base + ".txt"
            src_img = os.path.join(dirPath, img)
            src_txt = os.path.join(dirPath, txt)

            if img in test_files:
                dest_img = os.path.join(output_dir, "images", "val", f"{dir}_{img}")
                dest_txt = os.path.join(output_dir, "labels", "val", f"{dir}_{txt}")
            else:
                dest_img = os.path.join(output_dir, "images", "train", f"{dir}_{img}")
                dest_txt = os.path.join(output_dir, "labels", "train", f"{dir}_{txt}")

            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_txt, dest_txt)


if __name__ == "__main__":
    main()
