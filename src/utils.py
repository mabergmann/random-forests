import statistics
import numpy as np


def log(text, verbose=False):
    if verbose:
        print(text)


def get_possible_values(dataset, attr):
    assert not isinstance(dataset[0][attr], float)
    divisions = set()
    for instance in dataset:
        attr_value = instance[attr]
        divisions.add(attr_value)

    return divisions


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def accuracy(label, gt):
    label = np.asarray(label)
    gt = np.asarray(gt)
    correct = np.sum(label == gt)
    acc = correct / len(gt)
    assert 0 <= acc <= 1
    return acc


def precision(label, gt):
    assert len(label) == len(gt)
    classes = set(gt)
    precisions = []
    label = np.asarray(label)
    gt = np.asarray(gt)
    for c in classes:  # Calculates precision for each class
        predicted_as_c = label == c
        if sum(predicted_as_c) == 0:
            continue
        tp = 0
        for l, g in zip(label[predicted_as_c], gt[predicted_as_c]):
            if l == g:
                tp += 1
        class_precision = tp / np.sum(predicted_as_c)
        assert 0 <= class_precision <= 1
        precisions.append(class_precision)

    return statistics.mean(precisions)


def recall(label, gt):
    classes = set(gt)
    recalls = []
    label = np.asarray(label)
    gt = np.asarray(gt)
    for c in classes:  # Calculates precision for each class
        relevant = gt == c
        tp = 0
        for l, g in zip(label[relevant], gt[relevant]):
            if l == g:
                tp += 1

        class_recall = tp / np.sum(relevant)
        assert 0 <= class_recall <= 1
        recalls.append(class_recall)

    return statistics.mean(recalls)


def f1_score(label, gt):
    prec = precision(label, gt)
    recll = recall(label, gt)

    score = statistics.median([prec, recll])
    assert 0 <= score <= 1
    return score
