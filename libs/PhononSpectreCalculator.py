from .helpers import *
import scipy as sc
import math
import cmath
import functools
import scipy.linalg as sl

class PhononSpectreCalculator:
    def __init__(self, positions, mod_vectors, masses, power_constants, wave_vectors, path, amount_of_steps, basis_vector):
        self.positions = positions
        self.mod_vectors = mod_vectors

        self.mod_functions = self._get_mas_mod_functions(masses)
        self.power_constants = power_constants
        self.wave_vectors = wave_vectors
        self.path = path
        self.amount_of_steps = amount_of_steps
        self.basis_vector = basis_vector
        self.eigens_buffer = []
        self.path = path
        self.groups = self._generate_groups()
        self.defect_mass_matrix = self._get_defect_mass_matrix().round(6)
        self.degenerate_values = []
        self.count_of_values = len(list(filter(None, masses))) * 3
        self.general_dynamic_matrix = []
        self._calculate()

    def _get_mas_mod_functions(self, masses):
        positions_length = self.positions.shape[0]
        f1 = sc.zeros([positions_length, positions_length], dtype=sc.csingle)

        for i in range(positions_length):
            for j in range(positions_length):
                f1[i][j] = 1/positions_length * cmath.exp(-1 * 1j * (self.positions[i][0] * self.mod_vectors[j][0] + self.positions[i][1] * self.mod_vectors[j][1] + self.positions[i][2] * self.mod_vectors[j][2]))
        return sc.dot(f1, masses)

    def _generate_groups(self):
        positions = []
        for position in self.positions:
            if not sc.all(position == [0, 0, 0]):
                positions.append(position)

        positions = list(map(cast_to_lemma, positions))

        temp = list()
        unique_positions = [x for x in positions if x not in temp and (temp.append(x) or True)]
        groups = list(map(permute, unique_positions))

        return sorted(groups, key=functools.cmp_to_key(lambda a, b: sorting_groups(a, b)))

    def _get_defect_mass_matrix(self):
        count_vectors = self.mod_vectors.shape[0]
        basis_vectors = self._get_basis_vectors()
        defect_matrix = sc.zeros((count_vectors, count_vectors), dtype=sc.csingle)

        for i in range(count_vectors):
            possible_vectors = sc.copy(self.mod_vectors)

            for j in range(count_vectors):
                current = self.mod_vectors[i] - self.mod_vectors[j]
                found_index = None
                for basis_vector in basis_vectors:
                    for index, possible_vector in enumerate(possible_vectors):
                        if sc.all(possible_vector == current + basis_vector):
                            found_index = index

                    if found_index is not None:
                        defect_matrix[i][j] = self.mod_functions[found_index]
                        possible_vectors[found_index] = None
                        break
        self.simple_defect_matrix = defect_matrix
        return sc.kron(defect_matrix, sc.diag([1,1,1]))

    def _get_basis_vectors(self):
        basis_vectors = {'000': [0,0,0]}
        basis_variants = permute(self.basis_vector)

        for i in range(5):
            current_vectors = basis_vectors.copy()
            for key in basis_vectors:
                for variant in basis_variants:
                    new_vector = [basis_vectors[key][0] + variant[0], basis_vectors[key][1] + variant[1], basis_vectors[key][2] + variant[2]]
                    current_vectors[concat(new_vector, '|')] = new_vector
            basis_vectors = current_vectors

        return sc.asarray(list(basis_vectors.values()))

    def _calculate(self):
        points = self.amount_of_steps - 1
        for i in range(self.amount_of_steps):
            if points == 0:
                modifier = 0
            else:
                modifier = i / points * self.path

            kx = get_in_range(modifier, self.wave_vectors[0])
            ky = get_in_range(modifier, self.wave_vectors[1])
            kz = get_in_range(modifier, self.wave_vectors[2])

            self.general_dynamic_matrix = self._get_general_dynamic_matrix(kx, ky, kz).round(6)
            eigen_values = sl.eigvals(self.general_dynamic_matrix.real, b=self.defect_mass_matrix.real).real
            valid_values = self._select_values(eigen_values, self.count_of_values)
            valid_values = sc.sqrt(valid_values).real / (2 * sc.pi * 3 * 10**10)

            self.eigens_buffer.append(valid_values)
            self.degenerate_values = self._get_degenerate_values(valid_values)

    def _get_general_dynamic_matrix(self, kx, ky, kz):
        count_vectors = self.mod_vectors.shape[0]
        count_components = 3

        row_position = 0
        col_position = 0

        general_matrix = sc.zeros((count_vectors * count_components, count_vectors * count_components), dtype=sc.csingle)
        for i in range(count_vectors):
            sum_dynamic_matrix = self._get_sum_dynamic_matrixes(kx + self.mod_vectors[i][0], ky + self.mod_vectors[i][1], kz + self.mod_vectors[i][2])

            for j in range(count_components):
                for k in range(count_components):
                    general_matrix[row_position][col_position + k] = sum_dynamic_matrix[j][k]
                row_position += 1
            col_position += count_components
        return general_matrix

    def _get_sum_dynamic_matrixes(self, kx, ky, kz):
        sum_matrixes = 0
        for i in range(len(self.groups)):
            sum_matrixes += self._get_dynamic_matrix(self.groups[i], self.power_constants[i], kx, ky, kz)
        return sum_matrixes

    def _get_dynamic_matrix(self, group_of_positions, power_constant, kx, ky, kz):
        dynamic_matrix = sc.zeros((3,3), dtype=sc.csingle)

        for i in range(3):
            for j in range(3):
                dynamic_matrix[i][j] = self._get_dynamic_value(group_of_positions, power_constant, kx, ky, kz, i, j)
        return dynamic_matrix

    def _get_dynamic_value(self, group_of_positions, power_constant, kx, ky, kz, os1, os2):
        vector_length = get_vector_length(group_of_positions[0])
        dynamic_value = 0

        for position in group_of_positions:
            dynamic_value += self._get_dynamic_step(position, vector_length, kx, ky, kz, os1, os2)
        return power_constant * dynamic_value

    @staticmethod
    def _get_dynamic_step(position, length, kx, ky, kz, os1, os2):
        return ((position[os1] * position[os2]) / length**2) * (1 - cmath.exp(1j * (kx * position[0] + ky * position[1] + kz * position[2])))

    @staticmethod
    def _select_values(values, desired_count):
        non_infinity_values = sc.extract(sc.logical_and(values != math.inf, values != -math.inf), values)
        selected_values = sc.zeros(desired_count)
        current_selected_index = 0

        for i in range(desired_count):
            min_diff = math.inf
            min_value = math.inf
            for j in range(non_infinity_values.size):
                if abs(non_infinity_values[j]) < min_diff:
                    min_value = j
                    min_diff = non_infinity_values[j]

            if min_value < math.inf:
                selected_values[current_selected_index] = non_infinity_values[min_value]
                current_selected_index += 1
                non_infinity_values[min_value] = math.inf
        return sorted(selected_values)

    @staticmethod
    def _get_degenerate_values(eigs):
        keys = []
        values = []

        for eig in eigs:
            if eig in keys:
                values[keys.index(eig)] += 1
            else:
                keys.append(eig)
                values.append(1)
        table = {}

        for i in range(len(keys)):
            if values[i] not in table:
                table[values[i]] = []
            table[values[i]].append(keys[i])
        return table
