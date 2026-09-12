import json
import os
import sys
import base64

labelme_version = "5.8.2"
classes = ["Flame", "Smoke"]


def vgg_to_labelme_shape(region):
    shape = {}
    region_shape = region.get("shape_attributes", {})
    region_attr = region.get("region_attributes", {})

    if region_shape.get("name") == "polyline":
        shape["label"] = classes[int(region_attr.get("Class"))]
        shape["points"] = list(
            zip(region_shape["all_points_x"], region_shape["all_points_y"])
        )
        shape["group_id"] = None
        shape["description"] = ""
        shape["shape_type"] = "polygon"
        shape["flags"] = {}
        shape["mask"] = None
    else:
        return None
    return shape


def convert_vgg_to_labelme(images_path, vgg_json_path):
    with open(vgg_json_path, "r") as f:
        vgg_data = json.load(f)

    img_metadata = vgg_data.get("_via_img_metadata")
    # VGG VIA 2.x/3.x: dict with keys as filenames or image IDs
    for key, value in img_metadata.items():
        image_filename = value.get("filename", key)
        shapes = []
        for region in value.get("regions", []):
            shape = vgg_to_labelme_shape(region)
            if shape:
                shapes.append(shape)

        image_path = images_path + "/" + image_filename
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        labelme_json = {
            "version": "5.8.2",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": image_data,
        }
        # Remove extension from image filename for output json
        base_name = os.path.splitext(image_filename)[0]
        output_path = os.path.join(images_path, f"{base_name}.json")
        with open(output_path, "w") as out_f:
            json.dump(labelme_json, out_f, indent=2)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    images_path = (
        "/media/HDD_1/Work/Projects/Watchtower/02_Flare/Dataset/3_Approved/VGG/11_01"
    )
    vgg_json_path = images_path + ".json"
    convert_vgg_to_labelme(images_path, vgg_json_path)
    print("----- Done -----")
