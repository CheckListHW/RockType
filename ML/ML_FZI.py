# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

plt.style.use('fivethirtyeight')

df = pd.read_excel("упрощенный датасет.xlsx")

c = 20
FZI_1 = np.array([])
GK = np.zeros((0, c * 2 + 1,))
for i in np.arange(c + 1, len(df.FZI.to_numpy()) - c + 1):
    FZI_1 = np.append(FZI_1, df.FZI.to_numpy()[i])
    a = np.reshape(df.GK.to_numpy()[i - c - 1:i + c], (1, c * 2 + 1))
    GK = np.concatenate((GK, a), axis=0)  # идет подготовка данных для обучения

GK = np.reshape(GK, (GK.shape[0], GK.shape[1], 1))

# модель обучения LSTM
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
    keras.callbacks.ModelCheckpoint(
        "model/LSTM_reg_1.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ), ]

history = model.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)
# модель обучения LSTM

# plt.figure(figsize=(16, 8))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

b_model = keras.models.load_model('model/LSTM_reg_1.h5')
d = x_test
TIMES = np.arange(0, len(d))
plt.figure(figsize=(8, 80))
b_model_predict_d = b_model.predict(d)

# plt.plot(b_model_predict, TIMES, '-', y_test, TIMES, '-')
# plt.plot(TIMES[l:], b_model.predict(x_test), '-', TIMES[l:], y_test, '-')
# plt.legend(['pred', 'test'], loc='upper left')
# plt.title('Оценка результатов')
# plt.xlabel('Time')
# plt.ylabel('Flowrate')
# plt.show()

d = x_train
TIMES = np.arange(0, len(d))
b_model_predict_x_test = b_model.predict(x_test)
b_model_predict_d = b_model.predict(d)

# plt.figure(figsize=(8, 80))
# plt.plot(b_model.predict(d), TIMES, '-', y_train, TIMES, '-')
# plt.plot(TIMES[l:], b_model_predict_x_test, '-', TIMES[l:], y_test, '-')
# plt.legend(['pred', 'test'], loc='upper left')
# plt.title('Оценка результатов')
# plt.xlabel('Time')
# plt.ylabel('Flowrate')
# plt.show()

c = b_model.predict(d)
b = y_train
new_c = []
for a in c:
    for aa in a:
        new_c.append(aa)

df2 = pd.DataFrame(data={'FZI': new_c, 'FZI_pred': b})
df2.to_excel('ML_FZI.xlsx')
corrMatrix = df2.corr()
corrMatrix  # проверка корреляции между эталоном и предсказанной FZI

df2 = pd.read_excel('ML_FZI.xlsx')
df2 = df2.drop(columns=['Unnamed: 0'])
a = pd.merge_asof(df2.sort_values('FZI'), df.sort_values('FZI'), on='FZI', direction='nearest')
b = a.sort_values(by='MD', ascending=True)
b = b.drop(columns=['FZI_pred'])
b.to_excel('ML_FZI + geof final.xlsx')  # сопоставление предсказанной FZI с глубиной(MD) и каротажом (GK)

