import json
import os

def convert_via_to_yolo(via_json_path, output_dir, image_dir):
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Определение ключа для аннотаций
    img_metadata_key = None
    for key in ["_via_img_metadata", "via_region_data"]:
        if key in via_data and isinstance(via_data[key], dict):
            img_metadata_key = key
            break

    if img_metadata_key is None:
        raise KeyError("Не удалось найти корректный ключ с аннотациями в JSON-файле.")

    for key, value in via_data[img_metadata_key].items():
        # Получение имени файла
        if "filename" not in value:
            continue  # Пропускаем элементы без имени файла
        filename = value["filename"]

        # Получение размеров изображения
        img_width, img_height = None, None
        if "regions" in value and len(value["regions"]) > 0:
            region = value["regions"][0]["shape_attributes"]
            img_width = region.get("width", 640)  # Используем заглушку, если ширина не указана
            img_height = region.get("height", 480)  # Используем заглушку, если высота не указана

        if img_width is None or img_height is None:
            print(f"Предупреждение: Размеры изображения для {filename} не определены. Используются значения по умолчанию (640x480).")
            img_width, img_height = 640, 480

        label_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(label_file, "w") as f:
            for region in value.get("regions", []):
                bbox = region["shape_attributes"]
                class_name = region["region_attributes"].get("class")

                # Маппинг классов
                class_map = {"cube": 0, "pipe": 1, "cylinder": 2}
                class_id = class_map.get(class_name, -1)
                if class_id == -1:
                    continue

                # Конвертация координат в формат YOLO
                x_center = (bbox["x"] + bbox["width"] / 2) / img_width
                y_center = (bbox["y"] + bbox["height"] / 2) / img_height
                width = bbox["width"] / img_width
                height = bbox["height"] / img_height

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Пример использования
via_json_path = "annotations.json"
output_dir = "labels"
image_dir = "images"
convert_via_to_yolo(via_json_path, output_dir, image_dir)