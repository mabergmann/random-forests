from DecisionTree import DecisionTree
from scipy import stats
import statistics
from tqdm import tqdm


class RandomForest:
    trees = []
    trained = False

    def train(self, dataset, m, n):
        # m = number of attributes.
        # n = number o trees
        self.trees = []

        for _ in range(n):
            bootstrap = dataset.bootstrap()
            t = DecisionTree()
            t.n_attr = m
            t.train(bootstrap)
            self.trees.append(t)

        self.trained = True

    def __call__(self, sample):
        assert self.trained

        votes = [t(sample) for t in self.trees]
        votes_classes = [v[0] for v in votes]

        mode = stats.mode(votes_classes)[0][0][0]

        correct_confidences = [v[1] for v in votes if v[0] == mode]

        return mode, statistics.mean(correct_confidences)