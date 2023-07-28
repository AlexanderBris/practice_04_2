# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:21:01 2023

@author: Alexander
"""

# подгрузка стандартных библиотек
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# подготавливаем данные
#------------------------------------------------------------------
num_words = 10000
max_review_len = 200

# загружаем данные в переменную df
cur_dir = os.getcwd()


train = pd.read_csv(cur_dir + "//" + 'yelp_review_polarity_csv/train.csv',
                    header=None,
                    names=['Class', 'Review'])
# выделяем данные для обучения
reviews = train['Review']
# Выделяем правильные ответы
y_train = train['Class'] - 1
# создаем токенизатор Keras
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews)
# Преобразуем отзывы Yelp в числовое представление
sequences = tokenizer.texts_to_sequences(reviews)
# ограничиваем длину отзывов
x_train = pad_sequences(sequences, maxlen=max_review_len)

# настраиваем и тренируем сеть
#------------------------------------------------------------------
model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_review_len))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_save_path = 'best_model_gru_batch512.h5'
checkpoint_callback = ModelCheckpoint(model_save_path,  monitor='val_accuracy',  save_best_only=True, verbose=1)

history = model.fit(x_train, y_train,  epochs=3, batch_size=512, validation_split=0.1, callbacks=[checkpoint_callback])

model.load_weights(model_save_path)

# вывод результата обучения
#------------------------------------------------------------------
plt.plot(history.history['accuracy'],  label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],  label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'],  label='обучающий наборе')
plt.plot(history.history['val_accuracy'],  label='проверочный наборе')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Работаем с тестовыми данными
#------------------------------------------------------------------

test = pd.read_csv(cur_dir + "//" + 'yelp_review_polarity_csv/test.csv',  header=None,  names=['Class', 'Review'])
test_sequences = tokenizer.texts_to_sequences(test['Review'])
x_test = pad_sequences(test_sequences, maxlen=max_review_len)
y_test = test['Class'] - 1
model.evaluate(x_test, y_test, verbose=1)


#------------------------------------------------------------------
#------------------------------------------------------------------
# классификация отзывов
test = pd.read_csv(cur_dir + "//" + 'yelp_review_full_csv/test.csv',  header=None,  names=['Class', 'Review'])


def complete_y_data(input):
    ans = np.zeros((len(input), 5))
    for i in range(len(input)):
        ans[i][input[i]] = 1
    return ans

# выделяем данные для обучения
reviews = train['Review']
# Выделяем правильные ответы
classes = train['Class'] - 1
y_train = complete_y_data(classes)

# создаем токенизатор Keras
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews)
# Преобразуем отзывы Yelp в числовое представление
sequences = tokenizer.texts_to_sequences(reviews)
# ограничиваем длину отзывов
x_train = pad_sequences(sequences, maxlen=max_review_len)
# # создаем нейронную сеть
model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_review_len))
model.add(GRU(128))
model.add(Dense(128,activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_save_path = 'best_model_gru_4class.h5'
checkpoint_callback = ModelCheckpoint(model_save_path,  monitor='val_accuracy',  save_best_only=True, verbose=1)

# вывод результата обучения
#------------------------------------------------------------------
plt.plot(history.history['accuracy'],  label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],  label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

model.load_weights(model_save_path)


# проверка на тестовой выборке
#------------------------------------------------------------------
test = pd.read_csv(cur_dir + "//" + 'yelp_review_full_csv/test.csv',  header=None,  names=['Class', 'Review'])
test_sequences = tokenizer.texts_to_sequences(test['Review'])
x_test = pad_sequences(test_sequences, maxlen=max_review_len)
classes_test = test['Class'] - 1
y_test = complete_y_data(classes_test)
model.evaluate(x_test, y_test, verbose=1)

