from libs.PositionGenerator import PositionGenerator
from libs.PhononSpectreCalculator import PhononSpectreCalculator
from libs.helpers import *
import math
import scipy as sc
import scipy.linalg as sl

generator = PositionGenerator(8,8)
generator.generate(PositionGenerator.OCK_LATTICE, PositionGenerator.GCK_LATTICE)
positions = unzip_positions(generator.get_generated())

print(positions)

mod_vectors = list(map(lambda item: [(item[0] * math.pi)/4, (item[1] * math.pi)/4, (item[2] * math.pi)/4], positions))

masses = [80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 64, 64, 64, 64, 64, 0, 0, 0, 32, 0, 32, 32, 32, 32, 0, 31]
steps = 2
path = math.pi/8
power_constants = [55, 2, 1, 37, 40, 36]
wave_vectors = [[path, 0], [path, 0], [path, 0]]

Rr8 = PhononSpectreCalculator(sc.asarray(positions), sc.asarray(mod_vectors), masses, power_constants, wave_vectors, path, steps, [math.pi, math.pi, 0])

print(Rr8.eigens_buffer[0])
