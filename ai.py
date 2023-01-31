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
import sklearn.metrics as metrics
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from db import Exam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Строка с алфавитом для поиска символа по индексу на выходе из нейросети
alphabet ='0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЦЪЫЬЭЮЯ,-'
numbers = '0123456789'
letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
extra = ',-'

# Словарь для выбора типа модели
chrs = {'l': letters, 'n': numbers}

exam = Exam()


class EgeModel:

    def __init__(self, type: str) -> None:
        self.type = type
        self.load()

    def load(self) -> None:
        if os.path.exists(f'keras_models/{self.type}_ege_model.h5'):
            self.model = keras.models.load_model(f'keras_models/{self.type}_ege_model.h5')
            print('Модель загружена из файла')
        else:
            print('Файл для загрузки не найден')
            self.train()

    def make_model(self) -> None:
        ''' Создает модель (тип 1) для распознования символов ЕГЭ'''
        self.model = Sequential()
        self.model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
        self.model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(chrs[self.type]), activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    def make_data(self):
        x = pd.read_csv(f'./csv/{self.type}x.csv') # чтение csv с данными для обучения
        x = x.drop(x.columns[[0]], axis=1) # удаление строки index
        x = x.values # чтение массива с данными
        y = pd.read_csv(f'./csv/{self.type}y.csv')['0'].values # чтение меток для обучения 

        X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.15) # разделение данных на train и test

        X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1)) # "выпрямление" данных
        X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(chrs[self.type]))

        y_train_cat = keras.utils.to_categorical(y_train, len(chrs[self.type])) # преобазование значения символа в форму выходного слоя нейросети
        y_test_cat = keras.utils.to_categorical(y_test, len(chrs[self.type]))

        return X_train, X_test, y_train_cat, y_test_cat

    def train(self):
        ''' Обучает модель для символов ЕГЭ '''
        print('Старт предобработки данных и обучения')
        t_start = time.time() # время старта

        X_train, X_test, y_train, y_test = self.make_data()

        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[learning_rate_reduction], batch_size=128, epochs=30) # обучение нейросети
        print(f"{self.type} Модель обучилась, dT (min):", (time.time() - t_start) / 60) # отчет о завершении и затраченное время в мин
        self.model.save(f'keras_models/{self.type}_ege_model3.h5')


    def model_info(self):
        print(self.model.summary())

    def confusion_matrix(self):
        _, X_test, _, y_test = self.make_data()
        y_pred = self.model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        confusion_matrix = metrics.confusion_matrix(y_true=y_test_labels, y_pred=y_pred_labels)
        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in chrs[self.type]],
                  columns = [i for i in chrs[self.type]])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.show()


    def ege_predict_symbol(self, img):
        ''' Возвращает 1 символ на фото '''
        img_arr = np.expand_dims(img, axis=0) 
        img_arr = 1 - img_arr / 255.0 # конвертация диапазона 0-255 в 0-1
        img_arr = img_arr.reshape((1, 28, 28, 1)) # "выпрямление" данных
        predict = self.model.predict([img_arr]) # вычисление результата
        # for symbol_i in range(len(predict[0])):
        #     print(f'Prediction for symbol "{chrs[self.type][symbol_i]}" is: {predict[0][symbol_i]}')
        result = np.argmax(predict, axis=1) # поиск символа в алфавите
        return chrs[self.type][result[0]]


    def letters_extract(self, img, out_size=28):
        ''' Выделяет и возвращает массив с символами на фото '''
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
        # cv2.imshow("Letters", output)
        # cv2.waitKey(0)    
        return letters


    def img_to_str(self, img):
        ''' Конвертирует символы на фотографии в строку '''
        letters = self.letters_extract(img=img) # получение массива букв с фото
        s_out = ""
        for i in range(len(letters)):
            s_out += self.ege_predict_img(letters[i][2]) # присоединение следующей буквы к итоговой строке
        return s_out


    def img_to_str(self, img):
        ''' Конвертирует символы на фотографии в строку '''
        letters = self.letters_extract(img=img) # получение массива букв с фото
        s_out = ""
        for i in range(len(letters)):
            s_out += self.ege_predict_symbol(letters[i][2]) # присоединение следующей буквы к итоговой строке
        return s_out


    async def check_blanks(self, img_urls: List[str]):
        for url in img_urls:
            answers = []
            img = cv2.imread(url)
            exam_id_area = img[300:450, 150:850]
            exam_id = self.img_to_str(exam_id_area)
            print('exam_id', exam_id)
            scores_data = exam.get_scores_data(exam_id)
            for i in range(1, len(scores_data) + 1):
                y = 300 * (i + 1)
                answer_area = img[y:y+150, 150:850]
                answer = self.img_to_str(answer_area)
                answers.append(answer)
            print(exam_id, answers)


if __name__ == "__main__":
    ai = EgeModel('l')
    # s_out = ai.img_to_str("photos/test.png")
    # print(s_out)
    ai.confusion_matrix()
