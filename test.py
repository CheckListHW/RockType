import math

import numpy as np
import pandas

from rocktype_method import SheetReader

def near_value(value, mass):
    delta = math.inf
    near_index = -1

    for index in range(len(mass)):
        if abs(mass[index] - value) < delta:
            near_index = index
            old_delta = delta
            delta = abs(mass[index] - value)
            if old_delta < delta:
                return near_index

    return near_index

def prepare_dataset_for_ml(valid_md, gis_md):
    dataset_for_ml = []
    for i in valid_md:
        dataset_for_ml.append(near_value(i, gis_md))

    return dataset_for_ml

def start_ml():
    gis_filename = 'C:/Users/kosac/PycharmProjects/winland_R35/data/gis.xlsx'
    petro_filename = '/rocktype_data.xlsx'
    wb_fes = SheetReader('fzi', petro_filename, 'A', 'B', 'C', 'D', 'E')
    valid_data_frame = pandas.read_excel(wb_fes.file_method_data)
    gis_data_frame = pandas.read_excel(gis_filename)

    x = prepare_dataset_for_ml(valid_data_frame['Глубина'], gis_data_frame['MD'])
    dataset_for_ml = gis_data_frame.iloc[x]
    x = dataset_for_ml.set_index(np.arange(len(dataset_for_ml.index)))
    x['FZI'] = valid_data_frame['FZI']


if __name__ == '__main__':
    start_ml()
