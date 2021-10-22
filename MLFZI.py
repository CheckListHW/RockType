import os
import numpy as np
import pandas as pd
import random as python_random
import tensorflow as tf


from matplotlib import pyplot as plt
from numpy import arange
from pandas import read_excel
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from numpy.random import seed
from math import inf

os.environ['BASE_DIR'] = os.path.dirname(__file__)
BASE_DIR = os.environ['BASE_DIR']

from rocktype_method import SheetReader

seed(42)
python_random.seed(42)
plt.style.use('fivethirtyeight')


class ML_FZI():
    def __init__(self, petro_filename=None, gis_filename=None, batch_size=128, on_finished=None):
        self.batch_size = (batch_size // 32) * 32
        self.finished = on_finished
        self.gis_filename = gis_filename
        self.petro_filename = petro_filename

    @staticmethod
    def near_value(value, mass):
        delta = inf
        near_index = -1

        for index in range(len(mass)):
            if abs(mass[index] - value) < delta:
                near_index = index
                old_delta = delta
                delta = abs(mass[index] - value)
                if old_delta < delta:
                    return near_index

        return near_index

    @staticmethod
    def prepare_dataset_for_ml(valid_md, gis_md):
        dataset_for_ml = []
        for i in valid_md:
            dataset_for_ml.append(ML_FZI.near_value(i, gis_md))

        return dataset_for_ml

    def get_ML_FZI_url(self, filename=''):
        return BASE_DIR+'/Files/{file}'.format(file=filename)

    def get_accurancy(self):
        if hasattr(self, 'accurancy'):
            return abs(self.accurancy)
        return None

    def prepare_train_data(self, df, cor_list, limit):
        data_array = self.prepare_data(df, cor_list, limit)
        h = limit + 1
        d = len(df.FZI.to_numpy()) - limit + 1
        FZI = df.FZI.to_numpy()[h:d]
        return data_array, FZI

    def prepare_data(self, df, cor_list, limit):
        for c in cor_list:
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
            df[[c]] = scaler.fit_transform(df[[c]])

        data_array = np.zeros((0, len(cor_list), limit * 2 + 1))
        h = limit + 1
        d = len(df.FZI.to_numpy()) - limit + 1
        for i in np.arange(limit + 1, len(df.FZI.to_numpy()) - limit + 1):
            arr = [np.reshape(df[cor].to_numpy()[i - limit - 1:i + limit], (1, limit * 2 + 1)) for cor in cor_list]
            a = np.reshape(np.array(arr), (len(cor_list), 1 + limit * 2))
            a = np.reshape(a, (1, len(cor_list), limit * 2 + 1))
            data_array = np.concatenate((data_array, a), axis=0)
        return data_array

    def predict_FZI(self, model, df, cor_list, limit):
        predictions = [None] * limit + [i[0] for i in model.predict(self.prepare_data(df, cor_list, limit))] + [None] * limit
        df['FZI_predictions'] = predictions
        cors = ['MD'] + cor_list + ['FZI_predictions']
        df[cors].to_excel(self.get_ML_FZI_url('FZI_predictions.xlsx'))

    def start(self):
        wb_fes = SheetReader('fzi', self.petro_filename, 'A', 'B', 'C', 'D', 'E')
        valid_data_frame = pd.read_excel(wb_fes.file_method_data)
        gis_data_frame = pd.read_excel(self.gis_filename)

        valid_rows = self.prepare_dataset_for_ml(valid_data_frame['Глубина'], gis_data_frame['MD'])
        dataset_for_ml = gis_data_frame.iloc[valid_rows]

        dataset_for_ml_reindex = dataset_for_ml.set_index(arange(len(dataset_for_ml.index)))
        dataset_for_ml_reindex['FZI'] = valid_data_frame['FZI']

        self.data_ml = self.get_ML_FZI_url('dataset_for_ml.xlsx')
        dataset_for_ml_reindex.to_excel(self.data_ml)

        df = read_excel(self.data_ml)
        df = df.drop(columns=['Unnamed: 0'])

        cor_list = ['GK', 'BK', 'RHOB', 'IK', 'DT', 'NGK']
        limit = 20
        data_array, FZI = self.prepare_train_data(df, cor_list, limit)

        x_train, x_test, y_train, y_test = train_test_split(data_array, FZI, test_size=0.15, random_state=42)

        tf.random.set_seed(42)
        model = Sequential()
        model.add(LSTM(15, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "model/LSTM_reg_1.h5", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ), ]

        history = model.fit(x_train, y_train, batch_size=16, epochs=500, callbacks=callbacks, validation_split=0.15)

        model.evaluate(x_test, y_test), model.evaluate(x_train, y_train)

        b_model = keras.models.load_model('model/LSTM_reg_1.h5')
        b_model.evaluate(x_test, y_test)
        b_model.evaluate(x_train, y_train)
        df = pd.read_excel("gis.xlsx", index_col=0)
        self.predict_FZI(model, df, cor_list, limit)

        # графики
        plt.figure(figsize=(16, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.get_ML_FZI_url('model_loss.png'))

        plt.figure(figsize=(8, 80))
        d = x_test
        TIMES = np.arange(0, len(d))
        plt.plot(model.predict(d), TIMES, '-', y_test, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов(y_test)')
        plt.xlabel('FZI')
        plt.ylabel('Count')
        plt.savefig(self.get_ML_FZI_url('Оценка_результатов.png'))

        plt.figure(figsize=(8, 80))
        d = x_train
        TIMES = np.arange(0, len(d))
        plt.plot(model.predict(d), TIMES, '-', y_train, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов')
        plt.xlabel('FZI')
        plt.ylabel('Count')
        plt.savefig(self.get_ML_FZI_url('Оценка_результатов(y_train).png'))

        if self.finished is not None:
            self.finished()


if __name__ == "__main__":
    o = ML_FZI(petro_filename='rocktype_data.xlsx', gis_filename='gis.xlsx')
    o.start()