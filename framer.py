from ultralytics import YOLO
import cv2
import os
import time, torch
from pathlib import Path


def save_cropped_objects(result, model, output_dir='frames'):
    """Сохраняет вырезанные объекты из результатов детекции"""
    # Создаем папку для сохранения
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Получаем оригинальное изображение и путь
    orig_img = result.orig_img
    file_path = Path(result.path)
    base_name = file_path.stem

    # Обрабатываем каждый обнаруженный объект
    for i, box in enumerate(result.boxes):
        # Координаты bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Название класса и уверенность
        class_id = int(box.cls)
        class_name = model.names[class_id]
        conf = box.conf.item()

        # Вырезаем область изображения
        crop = orig_img[y1:y2, x1:x2]

        # Формируем имя файла
        save_name = f"{base_name}_{class_name}_{i}.jpg"
        save_path = os.path.join(output_dir, save_name)

        # Сохраняем фрагмент
        if crop.size > 0:
            cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


# Основной код
if __name__ == "__main__":
    # Конфигурация
    config = {
        'model_weights': 'runs/detect/train11/weights/best.pt' ,
        'input_folder': 'my_dataset_yolov5/images/test',
        'output_folder': 'detection_results',
        'conf_threshold': 0.5,
        'iou_threshold': 0.45,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }

    # Инициализация модели
    model = YOLO(config['model_weights']).to(config['device'])

    # Обработка изображений
    results = model.predict(
        source=os.path.join(config['input_folder'], '*'),
        conf=config['conf_threshold'],
        iou=config['iou_threshold'],
        save=True,
        project=config['output_folder'],
        exist_ok=True,
        show=False
    )

    # Сохранение фрагментов для каждого результата
    for result in results:
        save_cropped_objects(result, model)

    print(f"\nФрагменты объектов сохранены в папку: {os.path.abspath('frames')}")