import os
import random as python_random
import numpy as np
import pandas as pd

os.environ['BASE_DIR'] = os.path.dirname(__file__)
BASE_DIR = os.environ['BASE_DIR']

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from math import inf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow import random

from rocktype_method import SheetReader

np.random.seed(42)
python_random.seed(42)
plt.style.use('fivethirtyeight')


class ML:
    def __init__(self, petro_filename=None, gis_filename=None, on_finished=None,
                 cor_list=None, method_name='lucia'):
        if cor_list is None:
            cor_list = ['GK', 'BK', 'RHOB', 'IK', 'DT', 'NGK']

        self.cor_list = list(cor_list)
        self.method_name = method_name
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
            dataset_for_ml.append(ML.near_value(i, gis_md))

        return dataset_for_ml

    def get_ML_url(self, filename=''):
        return BASE_DIR + '/Files/{file}'.format(file=filename)

    def get_accurancy(self):
        if hasattr(self, 'accurancy'):
            return abs(self.accurancy)
        return None

    def prepare_train_data(self, df, cor_list, limit):
        data_array = self.prepare_data(df, cor_list, limit)
        h = limit + 1
        d = len(df.fzi.to_numpy()) - limit + 1
        fzi = df.fzi.to_numpy()[h:d]
        return data_array, fzi

    def prepare_data(self, df, cor_list, limit):
        for c in cor_list:
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
            df[[c]] = scaler.fit_transform(df[[c]])

        data_array = np.zeros((0, len(cor_list), limit * 2 + 1))
        h = limit + 1
        d = len(df.fzi.to_numpy()) - limit + 1
        for i in np.arange(limit + 1, len(df.fzi.to_numpy()) - limit + 1):
            arr = [np.reshape(df[cor].to_numpy()[i - limit - 1:i + limit], (1, limit * 2 + 1)) for cor in cor_list]
            a = np.reshape(np.array(arr), (len(cor_list), 1 + limit * 2))
            a = np.reshape(a, (1, len(cor_list), limit * 2 + 1))
            data_array = np.concatenate((data_array, a), axis=0)
        return data_array

    def predict_fzi(self, model, df, cor_list, limit):
        predictions = [None] * limit + [i[0] for i in model.predict(self.prepare_data(df, cor_list, limit))] + [
            None] * limit
        predictions_name = '{method_name}_predictions'.format(method_name=self.method_name)
        df[predictions_name] = predictions
        cors = ['MD'] + cor_list + [predictions_name]
        df[cors].to_excel(self.get_ML_url(predictions_name + '.xlsx'))

    def get_ml_data(self):
        wb_fes = SheetReader(self.method_name, self.petro_filename, 'A', 'B', 'C', 'D', 'E')

        valid_data_frame = pd.read_excel(wb_fes.file_method_data)
        gis_data_frame = pd.read_excel(self.gis_filename)

        valid_rows = self.prepare_dataset_for_ml(valid_data_frame['Глубина'], gis_data_frame['MD'])
        dataset_for_ml = gis_data_frame.iloc[valid_rows]

        dataset_for_ml_reindex = dataset_for_ml.set_index(np.arange(len(dataset_for_ml.index)))
        dataset_for_ml_reindex[self.method_name] = valid_data_frame[self.method_name]
        dataset_for_ml_reindex['fzi'] = dataset_for_ml_reindex[self.method_name]
        self.data_ml = self.get_ML_url('dataset_for_ml.xlsx')
        dataset_for_ml_reindex.to_excel(self.data_ml)

        return self.data_ml

    def start(self, train_filename=None):
        if train_filename is None:
            train_filename = self.get_ml_data()
        df = pd.read_excel(train_filename)
        df.drop(columns=['Unnamed: 0'])

        limit = 20
        data_array, fzi = self.prepare_train_data(df, self.cor_list, limit)

        x_train, x_test, y_train, y_test = train_test_split(data_array, fzi, test_size=0.15, random_state=42)

        random.set_seed(42)
        model = Sequential()
        model.add(LSTM(15, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        callbacks = [
            ModelCheckpoint(
                "model/LSTM_reg_1.h5", save_best_only=True, monitor="val_loss"
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ), ]

        history = model.fit(x_train, y_train, batch_size=16, epochs=500, callbacks=callbacks, validation_split=0.15)

        model.evaluate(x_test, y_test), model.evaluate(x_train, y_train)

        b_model = load_model('model/LSTM_reg_1.h5')
        b_model.evaluate(x_test, y_test)
        b_model.evaluate(x_train, y_train)
        df = pd.read_excel(self.gis_filename)
        df['fzi'] = pd.Series('', index=df.index)
        self.predict_fzi(model, df, self.cor_list, limit)

        # графики
        plt.figure(figsize=(16, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.get_ML_url('model_loss.png'))

        plt.figure(figsize=(8, 80))
        d = x_test
        TIMES = np.arange(0, len(d))
        plt.plot(model.predict(d), TIMES, '-', y_test, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов(y_test)')
        plt.xlabel('fzi')
        plt.ylabel('Count')
        plt.savefig(self.get_ML_url('Оценка_результатов.png'))

        plt.figure(figsize=(8, 80))
        d = x_train
        TIMES = np.arange(0, len(d))
        plt.plot(model.predict(d), TIMES, '-', y_train, TIMES, '-')
        plt.legend(['pred', 'test'], loc='upper left')
        plt.title('Оценка результатов')
        plt.xlabel('fzi')
        plt.ylabel('Count')
        plt.savefig(self.get_ML_url('Оценка_результатов(y_train).png'))

        if self.finished is not None:
            self.finished()


if __name__ == "__main__":
    o = ML(petro_filename='data/rocktype_data.xlsx', gis_filename='data/gis.xlsx')
    o.start(train_filename='data/сводный датасет.xlsx')
