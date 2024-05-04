###############################################################################
# Имя файла: augmentation.py
# Версия: 0.1.5
# Дата: 23-03-2024
# Разработчик: Артем Подлегаев
# Школа: МАОУ "СОШ № 14"
# Класс: 8 "Е"
# Авторское право: Артему Подлегаеву
# Описание: Практическая работа на конкурс Большие вызовы (аугментация)
###############################################################################

from augment.geometric import crop, rotate, scale, shear, translate
from augment.photometric import brightness_contrast, colorSpace, addNoise


# Функции геометрической трансформации
##############################################################################

# Функция обрезки изображения
def crop_transformations(image):
    return crop(image, point1=(100, 100), point2=(450, 400))

# Функция вращения изображения
def rotate_transformations(image):
    return rotate(image, angle=15, keep_resolution=True)

# Функция масштаба изображения
def scale_transformations(image):
    return scale(image, fx=1.5, fy=1.5, keep_resolution=False)

# Функция сдвига с поворотом изображения
def shear_transformations(image):
    return shear(image, shear_val=0.2, axis=1)

# Функция перемещения изображения
def translate_transformations(image):
    return translate(image, tx=50, ty=60)

# Функции фотометрической трансформации
##############################################################################

# Функция изменения контрастности и яркости
def b_contrast(image, alpha = 1.3, beta = 5):
    return brightness_contrast(image, alpha, beta)

# Функция изменения цветового пространства colorspace = 'hsv', 'ycrcb' 'lab'
def color_space(image, colorspace = 'hsv'):
    return colorSpace(image, colorspace)

# Функция внесения шума по Гауссу
def addNoise_gaussian(image, mean = 0, var = 0.08):
    return addNoise(image, 'gaussian', mean, var)

# Функция внесения шума "перец"
def addNoise_salt_pepper(image, sp_ratio = 0.5, noise_amount = 0.1):
    return addNoise(image, 'salt_pepper', sp_ratio, noise_amount)

# Функция внесения шума по Пуассону
def addNoise_poisson(image, noise_amount = 0.5):
    return addNoise(image, 'poisson', noise_amount)