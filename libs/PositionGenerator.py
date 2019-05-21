import numpy as np
import functools
import math
from .helpers import *

class PositionGenerator:
    GCK_LATTICE = 1
    OCK_LATTICE = 2
    PKR_LATTICE = 3

    def __init__(self, modifier1, modifier2):
        self.modifier1 = modifier1
        self.modifier2 = modifier2
        self.start = PositionGenerator.GCK_LATTICE
        self.end = PositionGenerator.GCK_LATTICE
        self.buffer = []


        self.step_size_mapper = {
            self.GCK_LATTICE: [1,1,0],
            self.OCK_LATTICE: [1,1,1],
            self.PKR_LATTICE: [1,0,0]
        }

        self.max_position_mapper = {
            self.GCK_LATTICE: [modifier2 / 2, modifier2 / 2, 0],
            self.OCK_LATTICE: [modifier2 / 2, modifier2 / 2, modifier2 / 2],
            self.PKR_LATTICE: [modifier2, 0, 0]
        }

        self.invariants_mapper = {
            self.GCK_LATTICE: permute([modifier2 / 2, modifier2 / 2, 0]) + permute([modifier2, modifier2, 0]) + permute([modifier2, modifier2 / 2, modifier2 / 2])
                              + permute([modifier2, 0, 0]) + permute([modifier2, modifier2, modifier2]),

            self.OCK_LATTICE: permute([modifier2 / 2, modifier2 / 2, modifier2 / 2]) + permute([modifier2, modifier2, modifier2])
                              + permute([modifier2, modifier2, 0]) + permute([modifier2, 0, 0]),

            self.PKR_LATTICE: permute([modifier2, 0, 0]) + permute([modifier2, modifier2, 0]) + permute([modifier2, modifier2, modifier2])
                            + permute([2 * modifier2, modifier2, modifier2]) + permute([2 * modifier2, 2 * modifier2, modifier2]) + permute([2 * modifier2, 2 * modifier2, 2 * modifier2])
                            + permute([2 * modifier2, 0, 0]) + permute([2 * modifier2, modifier2, 0])
        }

        self.step = self.step_size_mapper[self.start]
        self.max_position = self.max_position_mapper[self.end]
        self.max_length = find_n2(self.max_position)
        self.invariants = self.invariants_mapper[self.end]

    def generate(self, start, end):
        self.start = start
        self.end = end
        self.buffer = [[[0,0,0]]]
        self.step = self.step_size_mapper[start]
        self.max_position = self.max_position_mapper[end]
        self.max_length = find_n2(self.max_position)
        self.invariants = self.invariants_mapper[end]

        queue = [self.step]
        current_index = 0
        current_element = queue[current_index]
        mutators = permute(self.step)
        current_length = find_n2(current_element)
        cache = []

        while current_length < self.max_length and in_limit(current_element, self.max_position):
            self.buffer.append(self._extract_invariants(permute(current_element)))

            for mutator in mutators:
                possible_position = np.add(current_element, mutator).tolist()
                length = find_n2(possible_position)

                if length > current_length:
                    lemma = cast_to_lemma(possible_position)
                    if lemma not in cache:
                        cache.append(lemma)

                        if not self._check_in_queue(queue, possible_position):
                            queue.append(possible_position)

            queue = sorted(queue, key=functools.cmp_to_key(lambda a, b : sorting_positions(a, b)))
            current_index += 1

            if current_index >= len(queue):
                current_length = float("inf")
            else:
                current_element = queue[current_index]
                current_length = find_n2(current_element)

    def _extract_invariants(self, possible_values):
        unique_values = []

        for item in possible_values:
            if not self._has_similar(self.invariants, unique_values, item):
                unique_values.append(item)
        return unique_values

    def _check_in_queue(self, queue, item):
        invariants = permute(item)
        for invariant in invariants:
            if self._has_similar(self.invariants, queue, invariant):
                return True
        return False

    @staticmethod
    def _has_similar(invariants, position_buffer, item):
        for invariant in invariants:
            if (np.add(item, invariant).tolist() in position_buffer) or (np.subtract(invariant, item).tolist() in position_buffer):
                return True
        return False

    def get_generated(self):
        return self.buffer