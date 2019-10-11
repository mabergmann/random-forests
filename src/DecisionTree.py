import utils
from information_gain import calculate_info_gain_numerical, calculate_info_gain_categorical
from scipy import stats
from random import sample
import math

class DecisionTree:
    is_leaf = False
    is_trained = False
    division = None
    n_attr = -1

    def train(self, dataset):
        all_attrs = dataset.header.copy()
        all_attrs.remove("class")
        self.n_attr = int(math.sqrt(len(all_attrs)))
        self.attrs_to_use = sample(all_attrs, min(self.n_attr, len(all_attrs)))
        # self.attrs_to_use = all_attrs

        self.attrs_to_use.append("class")

        gain = self.find_attribute_with_most_information_gain(dataset)

        # if not gain == 0:
            # print(f"Attr = {self.div_attr}, div = {self.division}, gain = {gain}")

        if gain == 0:
            self.is_leaf = True

        elif self.is_numerical():
            dataset_smaller, dataset_bigger = dataset.filter_dataset_numerical(self.div_attr, self.division)
            dataset_smaller = dataset_smaller.remove_attribute(self.div_attr)
            dataset_bigger = dataset_bigger.remove_attribute(self.div_attr)
            self.forward = {'smaller': DecisionTree(), 'bigger': DecisionTree()}
            self.forward['smaller'].n_attr = self.n_attr
            self.forward['bigger'].n_attr = self.n_attr
            self.forward['smaller'].train(dataset_smaller)
            self.forward['bigger'].train(dataset_bigger)

        else:  # Categorical and not leaf
            valid_classes = utils.get_possible_values(dataset, self.div_attr)
            self.forward = {}
            for c in valid_classes:
                # print(f"{c}:")
                self.forward[c] = DecisionTree()
                self.forward[c].n_attr = self.n_attr
                new_dataset = dataset.filter_dataset_categorical(self.div_attr, c)
                new_dataset = new_dataset.remove_attribute(self.div_attr)
                self.forward[c].train(new_dataset)

        classes = [x['class'] for x in dataset]
        self.predicted_class = stats.mode(classes)[0]
        self.probability = classes.count(self.predicted_class) / len(dataset)

        # print(f"class = {self.predicted_class}")

        self.is_trained = True

    def find_attribute_with_most_information_gain(self, dataset):
        # Returns the gain

        best_attribute_gain = 0
        for attr in self.attrs_to_use:
            if attr == 'class':
                continue
            if isinstance(dataset[0][attr], float):
                gain, division = calculate_info_gain_numerical(dataset, attr)
            else:
                gain = calculate_info_gain_categorical(dataset, attr)
                division = None

            if gain > best_attribute_gain:
                self.div_attr = attr
                best_attribute_gain = gain
                self.division = division

            utils.log(f"Information gain for {attr} = {gain}")

        return best_attribute_gain

    def is_numerical(self):
        return self.division is not None

    def __call__(self, sample, *args, **kwargs):
        if not self.is_trained:
            raise Exception("Decision tree not trained yet. Please call dt.tain() before dt()")
        if self.is_leaf:
            return self.predicted_class, self.probability

        assert self.div_attr is not None

        if self.is_numerical():
            if sample[self.div_attr] < self.division:
                return self.forward['smaller'](sample)
            else:
                return self.forward['bigger'](sample)
        else:
            attr_class = sample[self.div_attr]
            if not attr_class in self.forward.keys():
                return self.predicted_class, self.probability
            else:
                return self.forward[attr_class](sample)