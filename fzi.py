import math

from SheetReader import SheetReader


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

        self.current_sw_column_name = c_sw
        self.residual_sw_column_name = r_sw

        self.calc_main(main_file)
        self.calculate_RTWS(main_file)

    def calculate_RTWS(self, sw_file):
        current_sw = SheetReader.get_column(sw_file, self.current_sw_column_name)
        residual_sw = SheetReader.get_column(sw_file, self.residual_sw_column_name)

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
        for x in range(len(x_pts_modify) - 1):
            rock_type_n = []

            for i in range(len(self.pron_por['Log(FZI)'])):
                if x_pts_modify[x] < self.pron_por['Log(FZI)'][i] < x_pts_modify[x + 1]:
                    rock_type_n.append([self.pron_por['Пористость'][i], self.pron_por['Проницаемость'][i]])

            dots_rock_type.append(rock_type_n)

        return dots_rock_type
