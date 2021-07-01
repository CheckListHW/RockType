import math

from SheetReader import SheetReader


class FZI:
    FZI_chart_x = []
    FZI_chart_y = []

    rock_type_borders = []
    dots_rock_type = []
    pron_por_fzi = []

    rock_type_chart_scale = 'log'
    RTWS_chart_type = 'current'

    rock_type_colors = []

    def __init__(self, fes_svod_file, sw_file, depth, porv, pron, layer, note):
        super().__init__()

        self.depth_column_name = depth
        self.porv_column_name = porv
        self.pron_column_name = pron
        self.layer_column_name = layer
        self.note_column_name = note

        self.calculate_chart_auto_FZI(fes_svod_file)
        self.calculate_RTWS(sw_file)

    def calculate_RTWS(self, sw_file):
        current_sw = SheetReader.get_column(sw_file, 'D')
        residual_sw = SheetReader.get_column(sw_file, 'V')

        self.modify_current_sw = []
        print(current_sw)
        for i in current_sw:
            if isinstance(i, float):
                self.modify_current_sw.append(i / 100)
        self.modify_current_sw = sorted(self.modify_current_sw)

        self.modify_residual_sw = []
        for i in residual_sw:
            if isinstance(i, float):
                self.modify_residual_sw.append(i / 100)
        self.modify_residual_sw = sorted(self.modify_residual_sw)

    def calculate_chart_auto_FZI(self, filename):
        wb_fes = SheetReader(filename, self.depth_column_name,
                             self.porv_column_name,
                             self.pron_column_name,
                             self.layer_column_name,
                             self.note_column_name, )

        self.FZI_chart_x = wb_fes.get_column_auto_FZI(['Log(FZI)'])
        self.FZI_chart_y = wb_fes.get_column_auto_FZI(['probability'])
        self.pron_por_fzi = wb_fes.get_column_auto_FZI(['Пористость', 'Проницаемость', 'Log(FZI)'])

    def calculate_rock_type(self, rock_type_borders):
        x_pts_modify = [-math.inf] + sorted(rock_type_borders)
        dots_rock_type = []
        for x in range(len(x_pts_modify) - 1):
            rock_type_n = []

            for i in range(len(self.pron_por_fzi['Log(FZI)'])):
                if x_pts_modify[x] < self.pron_por_fzi['Log(FZI)'][i] < x_pts_modify[x + 1]:
                    rock_type_n.append([self.pron_por_fzi['Пористость'][i], self.pron_por_fzi['Проницаемость'][i]])

            dots_rock_type.append(rock_type_n)

        return dots_rock_type
