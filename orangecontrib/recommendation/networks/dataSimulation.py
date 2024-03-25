import random
import numpy as np


def iou(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def get_random():
    return set(random.sample(range(0, 41), 5))


def matrix_of_kids_and_cartooons(values):
    return [[int(i in child) for i in range(40)] for child in values]


children = [get_random()]

for i in range(11):
    while True:
        current = get_random()
        for existing in children:
            jaccard = iou(current, existing)
            if jaccard == 0 or jaccard > 0.4:
                break
        else:
            children.append(current)
            break

for i in range(11):
    while True:
        current = get_random()
        for existing in children:
            jaccard = iou(current, existing)
            if 0.6 < jaccard < 0.9:
                break
        else:
            children.append(current)
            break

matrix_of_all_values = matrix_of_kids_and_cartooons(children)

rows = matrix_of_all_values

np.savetxt("inputData.csv", rows, delimiter=";", fmt='%d')
