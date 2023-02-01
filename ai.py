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
chrs = [numbers, letters]

exam = Exam()


class EgeModel:

    def __init__(self) -> None:
        self.models = [None, None]
        self.load()

    def load(self) -> None:
        if os.path.exists(f'keras_models/l_ege_model.h5'):
            self.models[1] = keras.models.load_model(f'keras_models/l_ege_model.h5')
            print('Модель l загружена из файла')
        else:
            print('Файл для загрузки l не найден')
            self.train(1)
        if os.path.exists(f'keras_models/n_ege_model.h5'):
            self.models[0] = keras.models.load_model(f'keras_models/n_ege_model.h5')
            print('Модель n загружена из файла')
        else:
            print('Файл для загрузки n не найден')
            self.train(0)

    def make_model(self, type) -> None:
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
        model0 = model
        model1 = model
        model0.add(Dense(len(chrs[0]), activation="softmax"))
        model1.add(Dense(len(chrs[1]), activation="softmax"))
        model0.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
        model1.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
        self.models = [model0, model1]


    def make_data(self, type):
        type_label = 'n' if type == 0 else 'l'
        x = pd.read_csv(f'./csv/{type_label}x.csv') # чтение csv с данными для обучения
        x = x.drop(x.columns[[0]], axis=1) # удаление строки index
        x = x.values # чтение массива с данными
        y = pd.read_csv(f'./csv/{type_label}y.csv')['0'].values # чтение меток для обучения 

        X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.15) # разделение данных на train и test

        X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1)) # "выпрямление" данных
        X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(chrs[type]))

        y_train_cat = keras.utils.to_categorical(y_train, len(chrs[type])) # преобазование значения символа в форму выходного слоя нейросети
        y_test_cat = keras.utils.to_categorical(y_test, len(chrs[type]))

        return X_train, X_test, y_train_cat, y_test_cat

    def train(self, type):
        ''' Обучает модель для символов ЕГЭ '''
        print('Старт предобработки данных и обучения')
        t_start = time.time() # время старта

        X_train, X_test, y_train, y_test = self.make_data(type)

        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        self.models[type].fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[learning_rate_reduction], batch_size=128, epochs=30) # обучение нейросети
        print(f"{type} Модель обучилась, dT (min):", (time.time() - t_start) / 60) # отчет о завершении и затраченное время в мин
        self.models[type].save(f'keras_models/{type}_ege_model3.h5')


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


    def ege_predict_symbol(self, img, type):
        ''' Возвращает 1 символ на фото '''
        img_arr = np.expand_dims(img, axis=0) 
        img_arr = 1 - img_arr / 255.0 # конвертация диапазона 0-255 в 0-1
        img_arr = img_arr.reshape((1, 28, 28, 1)) # "выпрямление" данных
        predict = self.models[type].predict([img_arr])
        # for symbol_i in range(len(predict[0])):
        #     print(f'Prediction for symbol "{chrs[self.type][symbol_i]}" is: {predict[0][symbol_i]}')
        result = np.argmax(predict, axis=1) # поиск символа в алфавите
        return chrs[type][result[0]]


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


    def img_to_str(self, img, type):
        ''' Конвертирует символы на фотографии в строку '''
        letters = self.letters_extract(img=img) # получение массива букв с фото
        s_out = ""
        for i in range(len(letters)):
            s_out += self.ege_predict_symbol(letters[i][2], type) # присоединение следующей буквы к итоговой строке
        return s_out

    
    def preprocess_blank(self, img):
        if True:
            return img
        else:
            raise TypeError


    def get_answers_from_blank(self, img):
        answers = []
        exam_id_area = img[300:450, 150:850]
        exam_id = self.img_to_str(exam_id_area, 0)
        try:
            scores_data = exam.get_scores_data(exam_id)
        except FileNotFoundError:
            print(f'Exam with ID {exam_id} not found in DB')
            return
        for i in range(len(scores_data)):
            y = 300 * (i + 2)
            answer_area = img[y:y+150, 150:850]
            answer = self.img_to_str(answer_area, scores_data[i][1])
            answers.append(answer)   
        return {'exam_id': exam_id, 'answers': answers}


    async def check_blanks(self, img_urls: List[str]):
        for url in img_urls:
            img = cv2.imread(url)
            try:
                img = self.preprocess_blank(img)
            except TypeError:
                print(f'Error occured, blank {url} is not checked')
                break
            blank_data = self.get_answers_from_blank(img)
            print(blank_data)


if __name__ == "__main__":
    # ai = EgeModel('l')
    # s_out = ai.img_to_str("photos/test.png")
    # print(s_out)
    # ai.confusion_matrix()
    pass
