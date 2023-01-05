import os
import cv2
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from typing import *
import time
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Строка с алфавитом для поиска символа по индексу на выходе из нейросети
alphabet ='0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЦЪЫЬЭЮЯ,-'
numbers = '0123456789'
letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
extra = ',-'

# Словарь для выбора типа модели
chrs = {'l': letters, 'n': numbers}


def ege_model_1(type: str):
    ''' Создает модель (тип 1) для распознования символов ЕГЭ'''
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(chrs[type]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def ege_model_2(type: str):
    ''' Создает модель (тип 1) для распознования символов ЕГЭ'''
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(chrs[type]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def ege_model_3(type: str):
    ''' Создает модель (тип 1) для распознования символов ЕГЭ'''
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(chrs[type]), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    return model


def ege_train(model: Any, type: str):
    ''' Обучает модель для символов ЕГЭ '''
    print('Старт предобработки данных и обучения')
    t_start = time.time() # время старта

    x = pd.read_csv(f'./csv/{type}x.csv') # чтение csv с данными для обучения
    x = x.drop(x.columns[[0]], axis=1) # удаление строки index
    x = x.values # чтение массива с данными
    y = pd.read_csv(f'./csv/{type}y.csv')['0'].values # чтение меток для обучения 

    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.15) # разделение данных на train и test

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1)) # "выпрямление" данных
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(chrs[type]))

    y_train_cat = keras.utils.to_categorical(y_train, len(chrs[type])) # преобазование значения символа в форму выходного слоя нейросети
    y_test_cat = keras.utils.to_categorical(y_test, len(chrs[type]))

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=128, epochs=30) # обучение нейросети
    print(f"{type} Модель обучилась, dT (min):", (time.time() - t_start) / 60) # отчет о завершении и затраченное время в мин


def ege_predict_img(model, img, type: str):
    ''' Возвращает 1 символ на фото '''
    img_arr = np.expand_dims(img, axis=0) 
    img_arr = 1 - img_arr / 255.0 # конвертация диапазона 0-255 в 0-1
    img_arr = img_arr.reshape((1, 28, 28, 1)) # "выпрямление" данных
    predict = model.predict([img_arr]) # вычисление результата
    result = np.argmax(predict, axis=1) # поиск символа в алфавите
    print(predict)
    print(max(predict[0]))
    return chrs[type][result[0]]


def letters_extract(image_file: str, out_size=28):
    ''' Выделяет и возвращает массив с символами на фото '''
    img = cv2.imread(image_file) # чтение фото
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # обесцвечивание 
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) # повыение контраста
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1) # утолшение символа

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # нахождение контуров символов

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))


    letters.sort(key=lambda x: x[0], reverse=False) # сортировка букв по Х координате
    cv2.imshow("Letters", output)
    cv2.waitKey(0)    
    return letters


def img_to_str(model: Any, image_file: str, type: str):
    ''' Конвертирует символы на фотографии в строку '''
    letters = letters_extract(image_file) # получение массива букв с фото
    s_out = ""
    for i in range(len(letters)):
        s_out += ege_predict_img(model, letters[i][2], type) # присоединение следующей буквы к итоговой строке
    return s_out


if __name__ == "__main__":
    type = 'l'
    # model = ege_model_3(type)
    # emnist_train(model, type)
    # model.save(f'keras_models/{type}_ege_model3.h5')

    model = keras.models.load_model(f'keras_models/{type}_ege_model3.h5')
    print('Model loaded')
    s_out = img_to_str(model, "photos/test.png", type)
    print(s_out)