import pandas as pd
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import numpy as np

# %matplotlib inline

import matplotlib.pyplot as plt

from ML.ML_FZI import x_train, x_test, y_test, y_train

plt.style.use('fivethirtyeight')

df = pd.read_excel("сводный датасет.xlsx")
df = df.drop(columns=['Unnamed: 0'])
df1 = df.drop(columns=['MD', 'FZI'])

c = 20
FZI_1 = np.array([])
GK = np.zeros((0, 6, c * 2 + 1))
for i in np.arange(c + 1, len(df.FZI.to_numpy()) - c + 1):
    FZI_1 = np.append(FZI_1, df.FZI.to_numpy()[i])
    a = np.reshape(np.array([[np.reshape(df.GK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                             [np.reshape(df.BK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                             [np.reshape(df.RHOB.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                             [np.reshape(df.IK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                             [np.reshape(df.DT.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))],
                             [np.reshape(df.NGK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))]]), (6, 1 + c * 2))
    a = np.reshape(a, (1, 6, 41))
    GK = np.concatenate((GK, a), axis=0)

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
    keras.callbacks.ModelCheckpoint(
        "model/LSTM_reg_1.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ), ]

history = model.fit(x_train, y_train, batch_size=128, epochs=18, callbacks=callbacks, validation_split=0.2)

plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

b_model = keras.models.load_model('model/LSTM_reg_1.h5')
plt.figure(figsize=(8, 80))

d = x_test
TIMES = np.arange(0, len(d))
plt.plot(b_model.predict(d), TIMES, '-', y_test, TIMES, '-')
# plt.plot(TIMES[l:], b_model.predict(x_test), '-', TIMES[l:], y_test, '-')
plt.legend(['pred', 'test'], loc='upper left')
plt.title('Оценка результатов')
plt.xlabel('Time')
plt.ylabel('Flowrate')
plt.show()

plt.figure(figsize=(8, 80))
d = x_train
TIMES = np.arange(0, len(d))
plt.plot(b_model.predict(d), TIMES, '-', y_train, TIMES, '-')
# plt.plot(TIMES[l:], b_model.predict(x_test), '-', TIMES[l:], y_test, '-')
plt.legend(['pred', 'test'], loc='upper left')
plt.title('Оценка результатов')
plt.xlabel('Time')
plt.ylabel('Flowrate')
plt.show()

c = np.concatenate(b_model.predict(d))
b = y_train
df2 = pd.DataFrame(c, columns=['FZI'])
df2 = pd.DataFrame(b, columns=['FZI_pred'])
df2['FZI'] = c
df2['FZI_pred'] = b
df2.to_excel('ML_FZI.xlsx')
corrMatrix = df2.corr()
corrMatrix

df2 = pd.read_excel('ML_FZI.xlsx')
df2 = df2.drop(columns=['Unnamed: 0'])
a = pd.merge_asof(df2.sort_values('FZI'), df.sort_values('FZI'), on='FZI', direction='nearest')
b = a.sort_values(by='MD', ascending=True)
b = b.drop(columns=['FZI'])
b.to_excel('ML_FZI + geof final.xlsx')
