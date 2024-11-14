from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
labels = {
    0: 'Unlabeled',
    1: 'Building',
    2: 'Fence',
    3: 'Other',
    4: 'Pedestrian',
    5: 'Pole',
    6: 'Roadline',
    7: 'Road',
    8: 'Sidewalk',
    9: 'Vegetation',
    10: 'Car',
    11: 'Wall',
    12: 'Traffic sign'
}
# Путь к файлу чекпоинта
best_checkpoint_dir = "/Users/sasa/PycharmProjects/pythonProject/checkpoint-1200"  # обновите путь
processor = SegformerImageProcessor.from_pretrained(best_checkpoint_dir)
model = SegformerForSemanticSegmentation.from_pretrained(best_checkpoint_dir)

# Путь к папке с изображениями и истинными масками, а также для сохранения предсказаний
input_folder = "/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/images"
true_mask_folder = "/Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/segmentation_masks"
output_folder = "/Users/sasa/PycharmProjects/pythonProject/segformer_outputs"
os.makedirs(output_folder, exist_ok=True)

# Функция для обработки изображения, расчета метрик и отображения
def process_image(image_path, true_mask_path, output_folder):
    image = Image.open(image_path).convert("RGB")

    # Предобработка изображения и получение предсказания
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Конвертируем маску в uint8 перед преобразованием в изображение
    predicted_mask_resized = Image.fromarray(predicted_mask.astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
    predicted_mask_resized_np = np.array(predicted_mask_resized)

    # Сохранение предсказанной маски
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace(".png", "_predicted.png"))
    predicted_mask_resized.save(output_path)

    # Если доступна истинная маска, рассчитать метрики
    metrics = {}
    if os.path.exists(true_mask_path):
        true_mask = Image.open(true_mask_path)
        true_mask = np.array(true_mask)[:, :, 0]

        # Инициализация метрик для каждого класса
        unique_classes = np.unique(true_mask)
        metrics['precision_per_class'] = []
        metrics['recall_per_class'] = []
        metrics['f1_per_class'] = []
        metrics['iou_per_class'] = []
        metrics['class_names'] = unique_classes

        for cls in unique_classes:
            true_binary = (true_mask == cls).flatten()
            pred_binary = (predicted_mask_resized_np == cls).flatten()

            metrics['precision_per_class'].append(precision_score(true_binary, pred_binary))
            metrics['recall_per_class'].append(recall_score(true_binary, pred_binary))
            metrics['f1_per_class'].append(f1_score(true_binary, pred_binary))
            metrics['iou_per_class'].append(jaccard_score(true_binary, pred_binary))

    # Возврат изображения, масок и метрик
    return image, true_mask, predicted_mask_resized, metrics

# Обработка 10 случайных изображений и отображение результатов
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
random_images = np.random.choice(image_files, 10, replace=False)

for idx, image_name in enumerate(random_images):
    image_path = os.path.join(input_folder, image_name)
    true_mask_path = os.path.join(true_mask_folder, image_name)

    # Обработка изображения
    image, true_mask, predicted_mask_resized, metrics = process_image(image_path, true_mask_path, output_folder)

    # Визуализация изображения и масок
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Изображение {idx+1}: {image_name}")

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Оригинальное изображение")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap="jet")
    plt.title("Истинная маска")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask_resized, cmap="jet")
    plt.title("Предсказанная маска")
    plt.axis("off")

    plt.show()
    # Вывод метрик
    print(f"\nРезультаты для изображения {image_name}:")
    for idx, cls in enumerate(metrics['class_names']):
        class_name = labels.get(cls)
        print(
            f"Класс {class_name}: Precision: {metrics['precision_per_class'][idx]:.4f}, Recall: {metrics['recall_per_class'][idx]:.4f}, F1-score: {metrics['f1_per_class'][idx]:.4f}, IoU: {metrics['iou_per_class'][idx]:.4f}"
        )

print("Все предсказанные маски успешно сохранены в папке:", output_folder)