from libs.PositionGenerator import PositionGenerator
from libs.PhononSpectreCalculator import PhononSpectreCalculator
from libs.ChartDrawer import ChartDrawer
from libs.helpers import *
import math
import scipy as sc
import scipy.linalg as sl

generator = PositionGenerator(2,2)
generator.generate(PositionGenerator.PKR_LATTICE, PositionGenerator.PKR_LATTICE)
positions = unzip_positions(generator.get_generated())

mod_vectors = list(map(lambda item: [item[0] * math.pi, item[1] * math.pi, item[2] * math.pi], positions))

masses = [137, 0, 0, 0, 16, 16, 16, 45]
steps = 25
path = math.pi
power_constants = [220, 110, 30]
wave_vectors = [[path, 0], [path, 0], [path, 0]]

Rr2 = PhononSpectreCalculator(sc.asarray(positions), sc.asarray(mod_vectors), masses, power_constants, wave_vectors, path, steps, [2 * math.pi, 0, 0])
drawer = ChartDrawer('charts')
drawer.draw('batio', Rr2)
