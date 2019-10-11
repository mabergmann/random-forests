import math
import statistics

from Datasets import CsvDataset
from RandomForest import RandomForest
import utils

datasets = [CsvDataset("../data/credit.csv"),
            CsvDataset("../data/vertebra-column.csv"),
            CsvDataset("../data/wine.csv")]

f = open("log.csv", 'w')
f.write("n_trees,dataset,acc,acc_sd,f1,f1_sd\n")

n_trees = 1
while n_trees <= 256:
    for dataset in datasets:
        print(f'dataset = {dataset.filename}\nNumber of trees = {n_trees}')
        accs = []
        f1s = []
        for n_fold in range(10):
            train_dataset, test_dataset = dataset.get_folds(n_fold)
            rf = RandomForest()

            rf.train(train_dataset, math.ceil(math.sqrt(len(dataset.header))), n_trees)

            predicted = list([rf(sample)[0] for sample in test_dataset])
            expected = list([sample['class'] for sample in test_dataset])

            acc = utils.accuracy(predicted, expected)
            f1 = utils.f1_score(predicted, expected)
            accs.append(acc)
            f1s.append(f1)

            print("Acc:", acc)
            print("F1 Score:", f1)

        print("Mean accuracy between folds:", statistics.mean(accs))
        print("Mean F1 score between folds:", statistics.mean(f1s))
        print("Standard deviation between accuracy:", statistics.stdev(accs))
        print("Standard deviation between F1 scores:", statistics.stdev(f1s))
        f.write(f"{n_trees},{dataset.filename},{statistics.mean(accs)},{statistics.stdev(accs)},{statistics.mean(f1s)},{statistics.stdev(f1s)}\n")

        print("\n", "#"*80, "\n")

    n_trees += n_trees

f.close()