import itertools
import math
import numpy as np

class PositionGenerator:
    GCK_LATTICE = 1
    OCK_LATTICE = 2
    PKR_LATTICE = 3

    def __init__(self, modifier):
        self.modifier = modifier
        self.start = PositionGenerator.GCK_LATTICE
        self.end = PositionGenerator.GCK_LATTICE
        self.buffer = []


        self.step_size_mapper = {
            self.GCK_LATTICE: [1,1,0],
            self.OCK_LATTICE: [1,1,1],
            self.PKR_LATTICE: [1,0,0]
        }

        self.max_position_mapper = {
            self.GCK_LATTICE: [modifier / 2, modifier / 2, 0],
            self.OCK_LATTICE: [modifier / 2, modifier / 2, modifier / 2],
            self.PKR_LATTICE: [modifier, 0, 0]
        }

        self.invariants_mapper = {
            self.GCK_LATTICE: np.concatenate([
                self._permute([modifier / 2, modifier / 2, 0]),
                self._permute([modifier, modifier, 0]),
                self._permute([modifier, modifier / 2, modifier / 2]),
                self._permute([modifier, 0, 0]),
                self._permute([modifier, modifier, modifier])
            ]),

            self.OCK_LATTICE: np.concatenate([
                self._permute([modifier / 2, modifier / 2, modifier / 2]),
                self._permute([modifier, modifier, modifier]),
                self._permute([modifier, modifier, 0]),
                self._permute([modifier, 0, 0])
            ]),

            self.PKR_LATTICE: np.concatenate([
                self._permute([modifier, 0, 0]),
                self._permute([modifier, modifier, 0]),
                self._permute([modifier, modifier, modifier]),
                self._permute([2 * modifier, modifier, modifier]),
                self._permute([2 * modifier, 2 * modifier, modifier]),
                self._permute([2 * modifier, 2 * modifier, 2 * modifier]),
                self._permute([2 * modifier, 0, 0]),
                self._permute([2 * modifier, modifier, 0])
            ])
        }

        self.step = self.step_size_mapper[self.start]
        self.max_position = self.max_position_mapper[self.end]
        self.max_length = self._find_n2(self.max_position)
        self.invariants = self.invariants_mapper[self.end]

    def generate(self, start, end):
        self.start = start
        self.end = end
        self.buffer = np.array([[[0,0,0]]])
        self.step = self.step_size_mapper[start]
        self.max_position = self.max_position_mapper[end]
        self.max_length = self._find_n2(self.max_position)
        self.invariants = self.invariants_mapper[end]

        queue = np.array([self.step])
        current_index = 0
        current_element = queue[current_index]
        mutators = self._permute(self.step)
        print('step', self.step, self._permute(self.step))

        current_length = self._find_n2(current_element)
        cache = np.array([])

        while current_length < self.max_length and self._in_limit(current_element, self.max_position):

            self.buffer += self._extract_invariants(self.invariants, self.max_position, self._permute(current_element))

            for mutator in mutators:
                print(current_element, mutator)
                possible_position = current_element + mutator
                length = self._find_n2(possible_position)

                if length > current_length:
                    lemma = self._cast_to_lemma(possible_position)
                    if lemma not in cache:
                        cache.append(lemma)

                        if not self._check_in_queue(queue, possible_position):
                            queue.append(possible_position)
            queue = sorted(queue, key=self._sorting_groups)
            current_index += 1

            if current_index > len(queue):
                current_length = math.inf
            else:
                current_element = queue[current_index]
                current_length = self._find_n2(current_element)

    def _find_n2(self, position):
        return position[0]**2 + position[1]**2 + position[2]**2

    def _permute(self, position):
        return np.array(list(set(
            itertools.chain(
                itertools.permutations(position),
                itertools.permutations([-position[0], position[1], position[2]]),
                itertools.permutations([position[0], -position[1], position[2]]),
                itertools.permutations([position[0], position[1], -position[2]]),
                itertools.permutations([-position[0], -position[1], position[2]]),
                itertools.permutations([-position[0], position[1], -position[2]]),
                itertools.permutations([position[0], -position[1], -position[2]]),
                itertools.permutations([-position[0], -position[1], -position[2]])
            )
        )))

    def _in_limit(self, value, max_limit):
        value_lemma = self._cast_to_lemma(value)
        max_limit_lemma = self._cast_to_lemma(max_limit)
        return self._concat(value_lemma) > self._concat(max_limit_lemma)

    def _concat(self, value):
        return str(value[0]) + str(value[1]) + str(value[2])

    def _extract_invariants(self, possible_values, max_value, l):
        unique_values = []

        for item in l:
            if not self._has_similar(possible_values, unique_values, item):
                unique_values.append(item)
        return unique_values

    def _check_in_queue(self, queue, item):
        invariants = self._permute(item)
        for invariant in invariants:
            if self._has_similar(self._invariants, queue, invariant):
                return True
        return False

    def _sorting_groups(self, item1, item2):
        n1 = self._find_n2(item1)
        n2 = self._find_n2(item2)

        if n1 == n2:
            return self._in_limit(item1, item2)
        else:
            return n1 < n2

    def _has_similar(self, invariants, position_buffer, item):
        for invariant in invariants:
            if (np.add(item, invariant).tolist() in position_buffer) or (np.subtract(invariant, item).tolist() in position_buffer):
                return True
        return False

    def _cast_to_lemma(self, position):
        lemma = [abs(x) for x in position]
        lemma.sort()
        return lemma

    def get_generated(self):
        return self.buffer