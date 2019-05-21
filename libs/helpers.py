import math
import scipy as sc
import itertools

def get_in_range(elem, range_list):
    if max(range_list) == range_list[1]:
        return min(range_list[1], elem)
    else:
        return max(range_list[1], range_list[0] - elem)

def get_vector_length(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def cast_to_lemma(position):
    lemma = [abs(x) for x in position]
    return sorted(lemma, reverse=True)

def permute(position):
    position.sort(reverse=True)
    permutations = unique(list(
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
    ))
    return list(map(list, permutations))

def unique(seq):  # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

def find_n2(position):
    return position[0] ** 2 + position[1] ** 2 + position[2] ** 2

def sorting_positions(position1, position2):
    n1 = find_n2(position1)
    n2 = find_n2(position2)

    if n1 == n2:
        return -1 if in_limit(position1, position2) else 1
    else:
        return n1 - n2

def sorting_groups(group1, group2):
    position1 = group1[0]
    position2 = group2[0]

    return sorting_positions(position1, position2)

def in_limit(value, max_limit):
    value_lemma = cast_to_lemma([int(x) for x in value])
    max_limit_lemma = cast_to_lemma([int(x) for x in max_limit])

    return concat(value_lemma) < concat(max_limit_lemma)

def concat(value, separator = ''):
    return str(value[0]) + separator + str(value[1]) + separator + str(value[2])

def unzip_positions(position_groups):
    positions = []
    for group in position_groups:
        for item in group:
            positions.append(item)
    return positions