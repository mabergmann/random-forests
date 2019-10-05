from scipy import stats
import numpy as np
import math


def info(classes):
    # Receives a list with the classes and calculate the entropy
    mode = stats.mode(classes)[0]
    pi = classes.count(mode) / len(classes)

    return -pi * math.log(pi, 2)


def info_numerical(dataset, attr, division):
    smaller_class = [x['class'] for x in dataset if x[attr] < division]
    bigger_class = [x['class'] for x in dataset if x[attr] > division]

    return (len(smaller_class) / len(dataset)) * info(smaller_class) + (
                len(bigger_class) / len(dataset)) * info(bigger_class)


def calculate_info_gain_numerical(dataset, attr):
    values = {x[attr] for x in dataset}  # Creates a set to remove duplicates
    values = list(values)
    if len(values) == 1:  # The attribute is pure. Makes no sense dividing it
        return 0, None

    divisions = [(x + y) / 2 for x, y in zip(values[:-1], values[1:])]
    info_gains = [info_numerical(dataset, attr, x) for x in divisions]

    best_division_idx = np.argmax(np.asarray(info_gains))
    best_division_value = divisions[best_division_idx]

    classes = [x['class'] for x in dataset]
    information = info(classes)

    smaller_class = [x['class'] for x in dataset if x[attr] < best_division_value]
    bigger_class = [x['class'] for x in dataset if x[attr] > best_division_value]

    information_a = (len(smaller_class) / len(dataset)) * info(smaller_class) + (
                len(bigger_class) / len(dataset)) * info(bigger_class)

    return information - information_a, best_division_value

def calculate_info_gain_categorical(dataset, attr):
    divisions = {}
    for instance in dataset:
        attr_value = instance[attr]
        if attr_value not in divisions.keys():
            divisions[attr_value] = [instance]
        else:
            divisions[attr_value].append(instance)

    d = len(dataset)  # Number of instance

    classes = [x['class'] for x in dataset]
    information = info(classes)

    information_a = 0
    for attr_value in divisions.keys():
        dj = len(divisions[attr_value])  # Number of instances in this class
        classes = [x['class'] for x in divisions[attr_value]]

        information_a += (dj / d) * info(classes)

    return information - information_a