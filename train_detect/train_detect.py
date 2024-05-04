###############################################################################
# Имя файла: train_detect.py
# Версия: 0.1.5
# Дата: 23-03-2024
# Разработчик: Артем Подлегаев
# Школа: МАОУ "СОШ № 14"
# Класс: 8 "Е"
# Авторское право: Артему Подлегаеву
# Описание: Практическая работа на конкурс Большие вызовы (обучение и детекция)
###############################################################################

import subprocess

# Параметры
yolo_dir = 'yolov5/'
dataset = 'dataset/data.yaml'
result_train_dir = 'yolov5/runs/train/exp'
result_detect_dir = 'yolov5/runs/detect/'

img_size = 640
batch = 9
epochs = 10
yolo_model = 'yolov5n.pt'

# Переключатель режимов работы программы обучение/детекция train - True, detect - False
mode_switch = True

# Функция обучения
def yolo_train(yolo_model, epochs, batch, img_size):
    train = (yolo_dir + 'train.py --img ' + str(img_size) + ' --batch ' + str(batch) + ' --epochs '
             + str(epochs) + ' --data ' + str(dataset) + ' --weights ' + str(yolo_model))
    print('Старт обучения...')

    # Запуск процесса обучения
    subprocess.call('python ' + train, shell=True)
    print('Обучение завершено')


# Функция детекции объектов
def yolo_detect(confidence, img_size):
    conf_thres = confidence
    detect = (yolo_dir + 'detect.py --img ' + str(img_size) + ' --weights ' + str(result_train_dir)
              + '/weights/best.pt --name result_data --save-txt --conf-thres '
              + str(conf_thres) + ' --source test_data')
    print('Старт детекции...')

    # Запуск процесса детекции
    subprocess.call('python ' + detect, shell=True)
    print('Детекция успешно заверщена.')

if(mode_switch):
    # Вызываем функцию обучения
    yolo_train(yolo_model, epochs, batch, img_size)
else:
    # Уровень точности детекции для сохранения результата
    confidence = 0.40
    # Вызываем функцию детектирования
    yolo_detect(confidence, img_size)
