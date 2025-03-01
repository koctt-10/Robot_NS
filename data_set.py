import json

# Загрузка аннотаций из VIA
with open('annotations.json', 'r') as f:
    via_data = json.load(f)

# Создание структуры COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "cube"},
        {"id": 2, "name": "pipe"},
        {"id": 3, "name": "cylinder"}
    ]
}

image_id = 1
annotation_id = 1

for key, value in via_data["_via_img_metadata"].items():
    # Информация об изображении
    image_info = {
        "id": image_id,
        "file_name": value["filename"],
        "width": value["size"][0],
        "height": value["size"][1]
    }
    coco_data["images"].append(image_info)

    # Информация об аннотациях
    for region in value["regions"]:
        bbox = region["shape_attributes"]
        category = region["region_attributes"]["class"]

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1 if category == "cube" else 2 if category == "pipe" else 3,
            "bbox": [bbox["x"], bbox["y"], bbox["width"], bbox["height"]],
            "area": bbox["width"] * bbox["height"],
            "iscrowd": 0
        }
        coco_data["annotations"].append(annotation)
        annotation_id += 1

    image_id += 1

# Сохранение в COCO формате
with open('annotations_coco.json', 'w') as f:
    json.dump(coco_data, f)