import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

# Загрузите предварительно обученную или дообученную модель YOLOv8
model = YOLO("/Users/sasa/PycharmProjects/pythonProject/best (4).pt")  # Укажите путь к обученной модели

# Цвет для класса "roadline" с полупрозрачностью
ROADLINE_COLOR = (255, 0, 0, 128)  # Красный цвет с альфа-каналом

def segment_image(image_path, mask_path):
    # Загрузка истинной маски и преобразование её в бинарный вид для класса "roadline" (6)
    gt_mask = plt.imread(mask_path) * 255
    gt_mask = np.where(gt_mask == 6, 1, 0).astype(np.uint8)  # Бинаризация для класса "roadline"
    gt_mask = gt_mask[:, :, 0]  # Избавляемся от лишнего канала, если он есть

    # Шаг 1: Загрузка изображения
    image = cv2.imread(image_path)
    original_image = image.copy()  # Копия для визуализации

    # Шаг 2: Сегментация изображения
    results = model.predict(source=image, task="segment")  # Используем сегментацию модели YOLOv8

    # Получаем сегментационные маски и классы
    masks = results[0].masks.data.cpu().numpy()  # Получаем массив масок из модели
    classes = results[0].boxes.cls.cpu().numpy()  # Получаем классы объектов из боксов

    # Шаг 3: Визуализация - наложение масок для класса "roadline"
    overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)  # Пустая маска с альфа-каналом
    pred_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Бинарная маска предсказания

    for mask, class_id in zip(masks, classes):
        if int(class_id) == 0:  # Если класс — "roadline"
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            overlay[mask_resized > 0.5] = ROADLINE_COLOR  # Применяем цвет маски
            pred_mask[mask_resized > 0.5] = 1  # Бинаризация предсказанной маски для класса "roadline"

    # Преобразование overlay в RGB
    overlay_rgb = overlay[:, :, :3]  # Убираем альфа-канал, оставляя только RGB
    segmented_image = cv2.addWeighted(overlay_rgb, 0.5, original_image, 0.5, 0)

    # Шаг 4: Сохранение результата
    output_path = Path("segmented_images") / (Path(image_path).stem + "_segmented.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), segmented_image)

    print(f"Сегментированное изображение сохранено в: {output_path}")

    # Преобразование масок в плоские массивы для метрик
    true_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    # Рассчитываем метрики
    iou = jaccard_score(true_mask_flat, pred_mask_flat)
    f1 = f1_score(true_mask_flat, pred_mask_flat)
    precision = precision_score(true_mask_flat, pred_mask_flat)
    recall = recall_score(true_mask_flat, pred_mask_flat)

    # Выводим результаты
    print(f"IoU: {iou:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return segmented_image

# Шаг 5: Запуск на примере изображений
example_images = ["/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/images/02_00_000.png", "/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/images/02_01_000.png"]
example_masks = ["/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/segmentation_masks/02_00_000.png", "/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/segmentation_masks/02_01_000.png"]
for i in range(len(example_images)):
    segmented_image = segment_image(example_images[i], example_masks[i])