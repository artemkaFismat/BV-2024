###############################################################################
# Имя файла: augmentation.py
# Версия: 0.1.1
# Дата: 14-02-2024
# Разработчик: Артем Подлегаев
# Школа: МАОУ "СОШ № 14"
# Класс: 8 "Е"
# Авторское право: Артему Подлегаеву
# Описание: Практическая работа на конкурс Большие вызовы (конвертер .xml в .txt (YOLO))
###############################################################################

import xmltodict
import os
from tqdm import tqdm

# Функция сохранения аннотаций в формате YOLO
def write_txt(data, save_path):
    with open(save_path, 'w') as f:
        for line in data:
            f.write(f"{line}\n")

# Функция конвертации .xml в .txt (xml_to_yolo)
def xml_to_yolo(file_path, save_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    annotation = xmltodict.parse(' '.join(data))
    img_h = int(annotation['annotation']['size']['height'])
    img_w = int(annotation['annotation']['size']['width'])
    drones = []
    if 'object' not in annotation['annotation'].keys():
        return
    if isinstance(annotation['annotation']['object'], dict):
        annotation['annotation']['object'] = [annotation['annotation']['object']]
    for obj in annotation['annotation']['object']:
        bbox = obj['bndbox']
        y_min, x_min, y_max, x_max = int(bbox['ymin']), int(bbox['xmin']), 
        int(bbox['ymax']), int(bbox['xmax'])
        w, h = x_max - x_min, y_max - y_min
        convert_x = x_min + w // 2
        convert_y = y_min + h // 2
        w /= img_w
        h /= img_h
        convert_x /= img_w
        convert_y /= img_h
        drones.append(f"0 {convert_x} {convert_y} {w} {h}")

    write_txt(drones, save_path)

# Конвертация тренировочного набора
labels_train_dir = 'dataset/labels/train'
annot_train_dir = 'Drone_TrainSet_XMLs'
os.makedirs(labels_train_dir, exist_ok=True)

for file in tqdm(os.listdir(annotation_train_dir)):
    xml_to_yolo(os.path.join(annotation_train_dir, file), 
    os.path.join(labels_train_dir, file.split('.')[0] + '.txt'))

# Конвертация валидационного набора
labels_val_dir = 'dataset/labels/val'
annotation_val_dir = 'Drone_TestSet_XMLs'
os.makedirs(labels_val_dir, exist_ok=True)

for file in tqdm(os.listdir(annotation_val_dir)):
    xml_to_yolo(os.path.join(annotation_val_dir, file),
    os.path.join(labels_val_dir, file.split('.')[0] + '.txt'))

