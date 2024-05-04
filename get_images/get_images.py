###############################################################################
# Имя файла: main.py
# Версия: 0.1.5
# Дата: 23-03-2024
# Разработчик: Артем Подлегаев
# Школа: МАОУ "СОШ № 14"
# Класс: 8 "Е"
# Авторское право: Артему Подлегаеву
# Описание: Практическая работа на конкурс Большие вызовы (создание dataset)
###############################################################################

import datetime
import random
import time
import urllib.request
from urllib.request import urlopen
import logging

import cv2
import numpy as np
from yamager import Yamager

from augmentation import *

yamager = Yamager()


# Функция получения случайного изображения через поисковую систему googl
def get_image_url():
    # Получаем случайное изображение через поиск google
    images_url = yamager.search_google_images(search)
    return random.choice(images_url)


# Функция получения случайного User Agent для имитации естественного запроса
def get_user_agents(filename):
    file_user_agents = open(filename, 'r')
    user_agents_list = file_user_agents.read()
    return user_agents_list.split('\n')


# Функция получения списка запросов для поисковой системы
def get_search_queries(filename):
    file_search_queries = open(filename, 'r')
    search_list = file_search_queries.read()
    return search_list.split('\n')


# Функция преобразования размера итогового изображения до 640x480 и сохранения
def image_save(image, path, filename, quality, size):
    image_out = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    write_status = cv2.imwrite(path + '/' + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if write_status is True:
        return True
    else:
        return False


# Функция получение изображения
def get_image(url, user_agent):
    req = urllib.request.Request(url, data=None, headers={'User-Agent': user_agent})
    req = urlopen(req)
    image = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image


# Получаем текущую дату
current_date = datetime.datetime.now().strftime('%d%m%Y')

# Получаем список User Agents (2000 вариантов)
user_agents_list = get_user_agents('user_agents.txt')

# Получаем список поисковых запросов для поиска изображений на заданную тему
search_list = get_search_queries('search_queries.txt')

# Подключение логов
logging.basicConfig(
    level=logging.INFO,
    filename = "get_images.log",
    format = "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt='%d.%m.%Y - %H:%M:%S',
    )

###############################################################################
# Параметры программы
###############################################################################
# Включение / выключние аугментации
get_augmentation = False

# Параметр качества сохраняемого изображения (0-100 %)
quality = 85

# Параметр размера сохраняемого изображения
# (для корректировки размера используется метод билинейной интерполяции)
size = [640, 480]

image_count = 1000

###############################################################################

iteration = 0
count = 0
count_all_images = 0
number_augment_modif_all = 0

while count <= image_count:
    number_augment_modif = 0

    # Получаем случайный поисковый запрос
    search = search_list[random.randint(0, len(search_list) - 1)]
    print('Запрос поисковой системе: ' + search)

    try:
        url = get_image_url()
        print('Обрабатывается адрес изображения: ' + url)

        user_agent = user_agents_list[random.randint(0, len(user_agents_list) - 1)]

        # Получаем исходное изображение
        image = get_image(url, user_agent)
        if (image_save(image, 'data', str(iteration) + str(current_date) + '.jpg', quality, size)):
            count += 1
            print('Загружено оригинальных изображений: ', count)
            print('---------------------------------------------')

            # Блок различных преобразований (аугментации)
            if (get_augmentation):
                image_save(crop_transformations(image), 'data', 'crop_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(rotate_transformations(image), 'data', 'rotate_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(scale_transformations(image), 'data', 'scale_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(shear_transformations(image), 'data', 'shear_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(translate_transformations(image), 'data', 'translate_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_save(b_contrast(image), 'data', 'b_co_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(b_contrast(image, 0.7, -3), 'data', 'b_c_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                #image_save(color_space(image), 'data', 'hsv_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                #number_augment_modif += 1
                #image_save(color_space(image, 'ycrcb'), 'data', 'ycrcb_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                #number_augment_modif += 1
                #image_save(color_space(image, 'lab'), 'data', 'lab_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                #number_augment_modif += 1
                image_save(addNoise_gaussian(image), 'data', 'gaussian_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(addNoise_salt_pepper(image), 'data', 'pepper_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(addNoise_poisson(image), 'data', 'poisson_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1

                #  Переключаем цветовое пространство в оттенки серого
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_save(b_contrast(image), 'data', 'b_contrast_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(b_contrast(image, 0.7, -10), 'data', 'b_cont_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1

        number_augment_modif_all += number_augment_modif
        count_all_images = count + number_augment_modif_all
        # Пишем лог
        logging.info('Порядковый номер изображения: ' + str(count))
        logging.info('URL: ' + url)
        logging.info('Число модификаций (аугментация): ' + str(number_augment_modif))
        logging.info('Результирующее число изображений в наборе: ' + str(count_all_images))
        print('Результирующее число изображений в наборе: ' + str(count_all_images))
        logging.info('--------------------------------------------------------------------')
        print('---------------------------------------------')

        # Для имитации реальных запросов устанавливаем случайное время паузы
        time_pause = random.randint(1, 25)
        time.sleep(time_pause)

        iteration += 1

    except Exception:
        continue
