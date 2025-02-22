import os, torch, time
from ultralytics import YOLO
# Конфигурация
model_weights = 'runs/detect/train/weights/best.pt'          # Путь к вашим весам
input_folder = 'Cans_marked/images/test'      # Папка с исходными изображениями
output_folder = 'digits_detection'# Папка для сохранения результатов
conf_threshold = 0.4              # Порог уверенности (0-1)
iou_threshold = 0.3              # Порог IoU для NMS (0-1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Автовыбор устройства

# Создаем выходную папку
os.makedirs(output_folder, exist_ok=True)

# Загрузка модели
model = YOLO(model_weights).to(device)

# Получаем список изображений
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(input_folder)
               if os.path.splitext(f)[1].lower() in image_extensions]

# Обработка изображений
start_time = time.time()

results = model.predict(
    source=os.path.join(input_folder, '*'),
    conf=conf_threshold,
    iou=iou_threshold,
    save=True,
    save_txt=True,          # Сохранять аннотации в txt
    save_conf=True,         # Сохранять уверенность в аннотациях
    project=output_folder,  # Путь для сохранения результатов
    name='',                # Подпапка не создается
    exist_ok=True,          # Перезаписывать существующие файлы
    line_width=2,           # Толщина линий
    show_labels=True,       # Показывать метки классов
    show_conf=True          # Показывать уверенность
)

# Статистика обработки
processing_time = time.time() - start_time
print(f'\nОбработано изображений: {len(image_files)}')
print(f'Общее время: {processing_time:.2f} сек')
print(f'Среднее время на изображение: {processing_time/len(image_files):.3f} сек')
print(f'Результаты сохранены в: {os.path.abspath(output_folder)}')