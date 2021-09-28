import numpy as np

from SheetReader import SheetReader
import data_poro as data


class lucia:
    def __init__(self, main_file, depth, porv, pron, layer, note, c_sw, r_sw):
        self.depth_column_name = depth
        self.porv_column_name = porv
        self.pron_column_name = pron
        self.layer_column_name = layer
        self.note_column_name = note

        self.current_sw_column_name = c_sw
        self.residual_sw_column_name = r_sw

        self.calc_main(main_file)
        self.calculate_RTWS(main_file)

        self.borders = [[data.poro_05, data.perm_05], [data.poro_15, data.perm_15],
                        [data.poro_25, data.perm_25], [data.poro_4, data.perm_4]]

    def calc_main(self, filename):
        wb_fes = SheetReader('lucia', filename, self.depth_column_name,
                             self.porv_column_name,
                             self.pron_column_name,
                             self.layer_column_name,
                             self.note_column_name)

        self.poro = wb_fes.get_column_method(['Пористость']).values.flatten()
        self.pron = wb_fes.get_column_method(['Проницаемость']).values.flatten()

    def calculate_RTWS(self, sw_file):
        current_sw = SheetReader.get_column(sw_file, 'D')
        residual_sw = SheetReader.get_column(sw_file, 'V')

        self.modify_current_sw = []
        for i in current_sw:
            if isinstance(i, float):
                self.modify_current_sw.append(i / 100)
        self.modify_current_sw = sorted(self.modify_current_sw)

        self.modify_residual_sw = []
        for i in residual_sw:
            if isinstance(i, float):
                self.modify_residual_sw.append(i / 100)
        self.modify_residual_sw = sorted(self.modify_residual_sw)

    def calc_rocktype(self):
        dots_rock_type = {'orange': [[], []], 'green': [[], []], 'blue': [[], []], 'red': [[], []], 'grey': [[], []]}

        for i in range(0, len(self.poro)):
            y1 = np.exp((22.56 - 12.08 * np.log(4.0)) + ((8.671 - 3.603 * np.log(4.0)) * np.log(self.poro[i])))
            if self.pron[i] < y1:
                dots_rock_type['orange'][0].append(self.poro[i])
                dots_rock_type['orange'][1].append(self.pron[i])
                continue

            y2 = np.exp((22.56 - 12.08 * np.log(2.5)) + ((8.671 - 3.603 * np.log(2.5)) * np.log(self.poro[i])))
            if self.pron[i] < y2:
                dots_rock_type['green'][0].append(self.poro[i])
                dots_rock_type['green'][1].append(self.pron[i])
                continue

            y3 = np.exp((22.56 - 12.08 * np.log(1.5)) + ((8.671 - 3.603 * np.log(1.5)) * np.log(self.poro[i])))
            if self.pron[i] < y3:
                dots_rock_type['blue'][0].append(self.poro[i])
                dots_rock_type['blue'][1].append(self.pron[i])
                continue

            y4 = np.exp((22.56 - 12.08 * np.log(0.5)) + ((8.671 - 3.603 * np.log(0.5)) * np.log(self.poro[i])))
            if self.pron[i] < y4:
                dots_rock_type['red'][0].append(self.poro[i])
                dots_rock_type['red'][1].append(self.pron[i])
                continue

            dots_rock_type['grey'][0].append(self.poro[i])
            dots_rock_type['grey'][1].append(self.pron[i])
        self.dots_rock_type = dots_rock_type
        return dots_rock_type
