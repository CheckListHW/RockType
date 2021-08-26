import math

import numpy as np

from SheetReader import SheetReader


def convert_float(value):
    try:
        return float(value)
    except ValueError as verr:
        return False
    except Exception as ex:
        return False


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

        self.current_sw = SheetReader.get_column(main_file, c_sw)
        self.residual_sw = SheetReader.get_column(main_file, r_sw)

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

    def rock_type_rtws(self):
        return self.rock_type_rtws

    def calc_rocktype(self, rock_type_borders):
        x_pts_modify = [-math.inf] + sorted(rock_type_borders)
        dots_rock_type = []
        self.modify_current_sw = []
        self.modify_residual_sw = []
        for x in range(len(x_pts_modify) - 1):
            rock_type_n = []
            modify_current_sw_n = []
            modify_residual_sw_n = []

            print(type(0.21))
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

            print(self.modify_current_sw)
            print(self.modify_residual_sw)

        return dots_rock_type

