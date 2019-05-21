import matplotlib.pyplot as plt
from libs.PhononSpectreCalculator import PhononSpectreCalculator

class ChartDrawer:
    def __init__(self, path: str):
        self.chartPath = path

    def draw(self, name: str, phonon_calculator: PhononSpectreCalculator):
        data = self._calculate_data(phonon_calculator)
        for line in data['y']:
            plt.plot(data['x'], line)

        plt.savefig(self.chartPath + '/' + name + '.png')

    def _calculate_data(self, phonon_calculator: PhononSpectreCalculator):
        points = len(phonon_calculator.eigens_buffer)
        x_axis = []
        y_axis = list(map(list, zip(*phonon_calculator.eigens_buffer)))

        for i in range(points):
            current_x = i / points * phonon_calculator.path
            x_axis.append(current_x)
        return {'x': x_axis, 'y': y_axis}