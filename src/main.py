import math
from Datasets import CsvDataset
from RandomForest import RandomForest
from tqdm import tqdm

dataset = CsvDataset("../data/credit.csv")
# dataset = CsvDataset("../data/spambase.csv")
# dataset = CsvDataset("../data/vertebra-column.csv")
# dataset = CsvDataset("../data/wine.csv")
rf = RandomForest()
rf.train(dataset, math.ceil(math.sqrt(len(dataset.header))), 10)

correct = 0
for i in tqdm(dataset):
    result = rf(i)
    if result[0] == i['class']:
        correct += 1

print("Correct:", correct)
print("Acc:", correct/len(dataset))
