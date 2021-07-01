import openpyxl
import pandas as pd
import numpy as np


class SheetReader:
    Url = None
    Columns = []
    Old_workbook = None
    Sheet = None
    FZI = openpyxl.Workbook()
    FZI_sheet = None
    auto_FZI = None

    def __init__(self, url, depth_column_name, porv_column_name, pron_column_name, layer_column_name, note_column_name):
        self.depth_column_name = depth_column_name
        self.pron_column_name = pron_column_name
        self.porv_column_name = porv_column_name
        self.layer_column_name = layer_column_name
        self.note_column_name = note_column_name
        self.Url = url
        self.Old_workbook = openpyxl.load_workbook(url)
        self.Sheet = self.Old_workbook['1']
        self.run()

    def run(self):
        self.pull_columns_to_new_workbook([self.depth_column_name, self.porv_column_name, self.pron_column_name])
        self.calculate()

    def pull_columns_to_new_workbook(self, columns):
        row_number_old_sheet = 2
        row_number_new_sheet = 2
        self.FZI_sheet = self.FZI.active
        # очень нужно не удаляй
        columns_name = ['Глубина', 'Пористость', 'Проницаемость']
        for i in range(len(columns_name)):
            cell = self.FZI_sheet.cell(row=1, column=i + 1)
            cell.value = columns_name[i]

        while (self.Sheet[columns[0] + str(row_number_old_sheet)]).value:

            if self._row_valid(row_number_old_sheet):
                for i in range(len(columns)):
                    cell = self.FZI_sheet.cell(row=row_number_new_sheet, column=i + 1)
                    cell.value = self.Sheet[columns[i] + str(row_number_old_sheet)].value
                row_number_new_sheet += 1
            row_number_old_sheet += 1

        self.FZI.save('Files/FZI.xlsx')

    def _row_valid(self, row_number: int) -> bool:
        if self.Sheet[self.depth_column_name + str(row_number)].value == \
                self.Sheet[self.depth_column_name + str(row_number - 1)].value:
            return False

        if str(self.Sheet[self.note_column_name + str(row_number)].value).__contains__('трещина'):
            return False

        if str(self.Sheet[self.porv_column_name + str(row_number)].value) == '0':
            return False
        if str(self.Sheet[self.porv_column_name + str(row_number)].value) == 'None':
            return False

        if str(self.Sheet[self.pron_column_name + str(row_number)].value) == '0':
            return False
        if str(self.Sheet[self.pron_column_name + str(row_number)].value) == 'None':
            return False

        for exception in ['ртинс', 'c1ok', 'c1bb', 'c1tl']:
            if self.Sheet[self.layer_column_name + str(row_number)].value.__contains__(exception):
                return False

        return True

    def calculate(self):
        df = pd.read_excel('Files/FZI.xLsx')
        df['Пористость'] = df['Пористость'] / 100
        df['porosity*'] = df['Пористость'] / (1 - df['Пористость'])
        df['корень'] = (df['Проницаемость'] / df['Пористость']) ** 0.5
        df['RQI'] = 0.0314 * df['корень']
        df['FZI'] = df['RQI'] / df['porosity*']
        df['Log(FZI)'] = np.log(df['FZI'])
        df['probability'] = 1 / (len(df))
        df = df.sort_values(by='Log(FZI)', ascending=True)  # сортировка данных по возрастанию
        df.to_excel('Files/auto_FZI.xlsx')  # функция сохранения в эксель

        self.auto_FZI = openpyxl.load_workbook('Files/auto_FZI.xlsx')
        auto_FZI_sheet = self.auto_FZI.active
        row_number = 3
        while auto_FZI_sheet['B' + str(row_number)].value:
            auto_FZI_sheet['J' + str(row_number)].value = \
                float(auto_FZI_sheet['J' + str(row_number - 1)].value) + \
                float(auto_FZI_sheet['J2'].value)
            row_number += 1
        self.auto_FZI.save('Files/auto_FZI.xlsx')

    @staticmethod
    def get_column(filename, column_name) -> []:
        df = openpyxl.load_workbook(filename).active
        column = []
        for i in range(len(df[column_name])):
            column.append(df[column_name + str(i+1)].value)
        return column

    def get_column_auto_FZI(self, column_name) -> []:
        df = pd.read_excel('Files/auto_FZI.xlsx')
        return df[column_name]
