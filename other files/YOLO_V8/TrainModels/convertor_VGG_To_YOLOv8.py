import os, json, cv2, shutil

jsonPath = "path/to/file.json"
imagePath = "path/to/images"
outputPath = "path/to/output"
img_type = ".jpg"
annot_type = ".txt"

if os.path.exists(outputPath):
    os.rmdir(outputPath)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Load the JSON data
with open(jsonPath, "r") as json_file:
    data = json.load(json_file)

# Iterate through image metadata
for image_id, metadata in data["_via_img_metadata"].items():
    filename = str(metadata["filename"]).replace(img_type, "")
    regions = metadata["regions"]

    if not len(regions):
        continue

    imgPath = imagePath + "/" + filename + img_type
    img = cv2.imread(imgPath)
    h, w, c = img.shape
    # Create a text file for the image
    with open(outputPath + "/" + filename + annot_type, "w") as text_file:
        for region in regions:
            class_id = [region["region_attributes"]["Class"]][0]
            all_points_x = region["shape_attributes"]["all_points_x"]
            all_points_y = region["shape_attributes"]["all_points_y"]

            # Create a line for the region
            r = ""
            for i in range(0, len(all_points_x)):
                r += " " + str(all_points_x[i] / w) + " " + str(all_points_y[i] / h)
            # line = f"{class_label} {' '.join(map(str, all_points_x))} {' '.join(map(str, all_points_y))}\n"
            line = class_id + r + "\n"
            text_file.write(line)

    shutil.copy(imgPath, outputPath + "/" + filename + img_type)
