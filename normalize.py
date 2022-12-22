import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2

alphabet ='0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЦЪЫЬЭЮЯ,-'
nums = [
    '0-0', '0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8', '0-9'
]
alphas = [
    '1-1A', '1-2B', '1-3V', '1-4G', '1-5D', '1-6E', '1-7YO', '1-8ZH', '1-9Z', '1-10I', '1-11YI', '1-12K', '1-13L', '1-14M', '1-15N', '1-16O', '1-17P',
    '1-18R', '1-19S', '1-20T', '1-21U', '1-22F', '1-23H', '1-24C', '1-25CH', '1-26SH', '1-27SCH', '1-28TZ', '1-29II', '1-30MZ', '1-31EA', '1-32YU', '1-33YA'
]
extra = [
    '2-1coma', '2-2minus'
]
alnums = nums + alphas


def normalize_numbers():
    for n in nums:
        print(n)
        files_dir = f'./datasets/{n}'
        files = os.listdir(files_dir)
        for file in files:
            img_dir = f'{files_dir}/{file}'
            with Image.open(img_dir) as img:
                img = img.crop((30, 30, img.height - 30, img.width - 30))
                img.thumbnail((28, 28))
                img.save(img_dir)


def normalize_letters():
    for l in alphas:
        print(l)
        files_dir = f'./datasets/{l}'
        files = os.listdir(files_dir)
        for file in files:
            img_dir = f'{files_dir}/{file}'
            with Image.open(img_dir) as img:
                new_img = Image.new('RGBA', img.size, 'WHITE')
                new_img.paste(img, (0, 0), img)
                new_img.convert('RGB')
                new_img.thumbnail((28, 28))
                new_img.save(img_dir)


def generate_y():
    y = []
    for ch in alnums:
        print(ch)
        y.extend([alnums.index(ch)] * len(os.listdir(f'./datasets/{ch}')))
    y = np.array(y)
    pd.DataFrame(y).to_csv('y.csv')


def generate_x():
    x = []
    for ch in alnums:
        print(ch)
        for file in os.listdir(f'./datasets/{ch}'):
            img = cv2.imread(f'./datasets/{ch}/{file}', 0)
            img = cv2.bitwise_not(img)
            img = img / 255.0
            arr = np.array(img)
            x.append(arr)
    x = np.array(x)
    pd.DataFrame(x).to_csv('x.csv')


for i in alnums:
    print(i, len(os.listdir(f'./datasets/{i}')))