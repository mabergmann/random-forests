from Datasets import CsvDataset
from DecisionTree import DecisionTree

dataset = CsvDataset("../data/credit.csv")
# dataset = CsvDataset("../data/spambase.csv")
# dataset = CsvDataset("../data/vertebra-column.csv")
# dataset = CsvDataset("../data/wine.csv")
dt = DecisionTree()
dt.train(dataset)


