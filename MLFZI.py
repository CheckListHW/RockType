import os
from math import inf
from os.path import dirname, isfile

import numpy as np
import pandas
from matplotlib import pyplot as plt
from numpy import array, zeros, reshape, append, arange, concatenate

from pandas import read_excel, merge_asof, DataFrame

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import tensorflow.keras.callbacks as keras_callbacks

from rocktype_method import SheetReader

BASE_DIR = os.environ['BASE_DIR']

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

    def start(self):
        wb_fes = SheetReader('fzi', self.petro_filename, 'A', 'B', 'C', 'D', 'E')
        valid_data_frame = pandas.read_excel(wb_fes.file_method_data)
        gis_data_frame = pandas.read_excel(self.gis_filename)

        valid_rows = self.prepare_dataset_for_ml(valid_data_frame['Глубина'], gis_data_frame['MD'])
        dataset_for_ml = gis_data_frame.iloc[valid_rows]

        dataset_for_ml_reindex = dataset_for_ml.set_index(arange(len(dataset_for_ml.index)))
        dataset_for_ml_reindex['FZI'] = valid_data_frame['FZI']

        self.data_ml = self.get_ML_FZI_url('dadaset_for_ml.xlsx')
        dataset_for_ml_reindex.to_excel(self.data_ml)

        df = read_excel(self.data_ml)
        df = df.drop(columns=['Unnamed: 0'])

        c = 20
        FZI_1 = array([])
        GK = zeros((0, 6, c * 2 + 1))
        for i in arange(c + 1, len(df.FZI.to_numpy()) - c + 1):
            FZI_1 = append(FZI_1, df.FZI.to_numpy()[i])
            a = reshape(array([[reshape(df.GK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                                     [reshape(df.BK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                                     [reshape(df.RHOB.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                                     [reshape(df.IK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                                     [reshape(df.DT.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                                     [reshape(df.NGK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))]]), (6, 1 + c * 2))
            a = reshape(a, (1, 6, 41))
            GK = concatenate((GK, a), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(GK, FZI_1, test_size=0.30, random_state=42)
        model = Sequential()
        model.add(LSTM(200, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(200, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Dense(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        callbacks = [
            keras_callbacks.ModelCheckpoint(
                "LSTM_reg_1.h5", save_best_only=True, monitor="val_loss"
            ),
            keras_callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ), ]

        # скорость обучения быстрее если batch_size больше
        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=18,
                            callbacks=callbacks, validation_split=0.2)

        b_model = load_model('LSTM_reg_1.h5')

        c = concatenate(b_model.predict(x_train))
        b = y_train

        df2 = DataFrame(data={'FZI': c, 'FZI_pred': b})
        df2.to_excel(self.get_ML_FZI_url('ML_FZI.xlsx'))
        self.accurancy = df2.corr()['FZI_pred']['FZI']

        df2 = read_excel(self.get_ML_FZI_url('ML_FZI.xlsx'))
        df2 = df2.drop(columns=['Unnamed: 0'])
        a = merge_asof(df2.sort_values('FZI'), df.sort_values('FZI'), on='FZI', direction='nearest')
        b = a.sort_values(by='MD', ascending=True)
        b = b.drop(columns=['FZI'])
        b.to_excel(self.get_ML_FZI_url('ML_FZI + geof final.xlsx'))


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
        plt.plot(b_model.predict(d), TIMES, '-', y_test, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов(y_test)')
        plt.xlabel('Time')
        plt.ylabel('Flowrate')
        plt.savefig(self.get_ML_FZI_url('Оценка_результатов.png'))

        plt.figure(figsize=(8, 80))
        d = x_train
        TIMES = np.arange(0, len(d))
        plt.plot(b_model.predict(d), TIMES, '-', y_train, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов')
        plt.xlabel('Time')
        plt.ylabel('Flowrate')
        plt.savefig(self.get_ML_FZI_url('Оценка_результатов(y_train).png'))

        if self.finished is not None:
            self.finished()

    def get_ML_FZI_url(self, filename=''):
        return BASE_DIR+'/Files/{file}'.format(file=filename)

    def get_accurancy(self):
        if hasattr(self, 'accurancy'):
            return abs(self.accurancy)
        return None

if __name__ == "__main__":
    o = ML_FZI()
    o.start()
