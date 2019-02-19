from libs.phononGeneration import PositionGenerator

generator = PositionGenerator(2)
generator.generate(PositionGenerator.GCK_LATTICE, PositionGenerator.PKR_LATTICE)
print(generator.get_generated())