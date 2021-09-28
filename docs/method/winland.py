import math

from numpy import log, exp

from SheetReader import SheetReader
import data_poro as data



def convert_float(value):
    try:
        return float(value)
    except ValueError as verr:
        return False
    except Exception as ex:
        return False

class winland:
    def __init__(self, fes_svod_file, depth, porv, pron, layer, note, c_sw, r_sw):
        self.depth_column_name = depth
        self.porv_column_name = porv
        self.pron_column_name = pron
        self.layer_column_name = layer
        self.note_column_name = note

        self.current_sw = SheetReader.get_column(fes_svod_file, c_sw)
        self.residual_sw = SheetReader.get_column(fes_svod_file, r_sw)

        self.calc_main(fes_svod_file)

    def calc_main(self, filename):
        wb_fes = SheetReader('winland', filename, self.depth_column_name,
                             self.porv_column_name,
                             self.pron_column_name,
                             self.layer_column_name,
                             self.note_column_name)

        self.probability = wb_fes.get_column_method(['probability'])
        self.method = wb_fes.get_column_method(['winland'])
        self.pron_por = wb_fes.get_column_method(['Пористость', 'Проницаемость', 'winland'])

    def calc_rocktype(self, rock_type_borders):
        x_pts_modify = [-math.inf] + sorted(rock_type_borders)
        dots_rock_type = []
        self.modify_current_sw = []
        self.modify_residual_sw = []
        for x in range(len(x_pts_modify) - 1):
            rock_type_n = []
            modify_current_sw_n = []
            modify_residual_sw_n = []

            for i in range(len(self.pron_por['winland'])):
                if x_pts_modify[x] < self.pron_por['winland'][i] < x_pts_modify[x + 1]:
                    rock_type_n.append([self.pron_por['Пористость'][i], self.pron_por['Проницаемость'][i]])
                    value_current_sw = convert_float(self.current_sw[i])
                    value_residual_sw = convert_float(self.residual_sw[i])
                    if value_current_sw:
                        modify_current_sw_n.append(value_current_sw)
                    if value_residual_sw:
                        modify_residual_sw_n.append(value_residual_sw)

            dots_rock_type.append(rock_type_n)
            self.modify_current_sw.append(modify_current_sw_n)
            self.modify_residual_sw.append(modify_residual_sw_n)
        return dots_rock_type


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
            y1 = log((22.56 - 12.08 * log(4.0)) + ((8.671 - 3.603 * log(4.0)) * log(self.poro[i])))
            if self.pron[i] < y1:
                dots_rock_type['orange'][0].append(self.poro[i])
                dots_rock_type['orange'][1].append(self.pron[i])
                continue

            y2 = log((22.56 - 12.08 * log(2.5)) + ((8.671 - 3.603 * log(2.5)) * log(self.poro[i])))
            if self.pron[i] < y2:
                dots_rock_type['green'][0].append(self.poro[i])
                dots_rock_type['green'][1].append(self.pron[i])
                continue

            y3 = log((22.56 - 12.08 * log(1.5)) + ((8.671 - 3.603 * log(1.5)) * log(self.poro[i])))
            if self.pron[i] < y3:
                dots_rock_type['blue'][0].append(self.poro[i])
                dots_rock_type['blue'][1].append(self.pron[i])
                continue

            y4 = log((22.56 - 12.08 * log(0.5)) + ((8.671 - 3.603 * log(0.5)) * log(self.poro[i])))
            if self.pron[i] < y4:
                dots_rock_type['red'][0].append(self.poro[i])
                dots_rock_type['red'][1].append(self.pron[i])
                continue

            dots_rock_type['grey'][0].append(self.poro[i])
            dots_rock_type['grey'][1].append(self.pron[i])
        self.dots_rock_type = dots_rock_type
        return dots_rock_type


class fzi:
    probability = []
    method = []

    rock_type_borders = []
    dots_rock_type = []
    pron_por = []

    rock_type_chart_scale = 'log'
    RTWS_chart_type = 'current'

    rock_type_colors = []

    def __init__(self, main_file, depth, porv, pron, layer, note, c_sw, r_sw):
        super().__init__()

        self.depth_column_name = depth
        self.porv_column_name = porv
        self.pron_column_name = pron
        self.layer_column_name = layer
        self.note_column_name = note


        self.calc_main(main_file)

    def calc_main(self, filename):
        wb_fes = SheetReader('fzi', filename, self.depth_column_name,
                             self.porv_column_name,
                             self.pron_column_name,
                             self.layer_column_name,
                             self.note_column_name, )

        self.probability = wb_fes.get_column_method(['probability'])
        self.method = wb_fes.get_column_method(['Log(FZI)'])
        self.pron_por = wb_fes.get_column_method(['Пористость', 'Проницаемость', 'Log(FZI)'])

    def calc_rocktype(self, rock_type_borders):
        x_pts_modify = [-math.inf] + sorted(rock_type_borders)
        dots_rock_type = []
        self.modify_current_sw = []
        self.modify_residual_sw = []
        for x in range(len(x_pts_modify) - 1):
            rock_type_n = []
            modify_current_sw_n = []
            modify_residual_sw_n = []

            for i in range(len(self.pron_por['Log(FZI)'])):
                if x_pts_modify[x] < self.pron_por['Log(FZI)'][i] < x_pts_modify[x + 1]:
                    rock_type_n.append([self.pron_por['Пористость'][i], self.pron_por['Проницаемость'][i]])
                    value_current_sw = convert_float(self.current_sw[i])
                    value_residual_sw = convert_float(self.residual_sw[i])
                    if value_current_sw:
                        modify_current_sw_n.append(value_current_sw)
                    if value_residual_sw:
                        modify_residual_sw_n.append(value_residual_sw)

            dots_rock_type.append(rock_type_n)
            self.modify_current_sw.append(modify_current_sw_n)
            self.modify_residual_sw.append(modify_residual_sw_n)
        return dots_rock_type