import utils
from information_gain import calculate_info_gain_numerical, calculate_info_gain_categorical
from scipy import stats


class DecisionTree:
    def __init__(self):
        self.is_leaf = False
        self.is_trained = False

    def train(self, dataset):
        gain = self.find_attribute_with_most_information_gain(dataset)

        if gain == 0:
            self.define_as_leaf(dataset)

        elif self.is_numerical():
            dataset_smaller, dataset_bigger = dataset.filter_dataset_numerical(self.div_attr, self.division)
            self.forward = {'smaller': DecisionTree(), 'bigger': DecisionTree()}
            self.forward['smaller'].train(dataset_smaller)
            self.forward['bigger'].train(dataset_bigger)

        else:  # Categorical and not leaf
            valid_classes = utils.get_possible_values(dataset, self.div_attr)
            self.forward = {}
            for c in valid_classes:
                self.forward[c] = DecisionTree()
                new_dataset = dataset.filter_dataset_categorical(self.div_attr, c)
                self.forward[c].train(new_dataset)

        self.is_trained = True

    def define_as_leaf(self, dataset):
        self.is_leaf = True
        classes = [x['class'] for x in dataset]
        self.predicted_class = stats.mode(classes)[0]
        self.probability = classes.count(self.predicted_class) / len(dataset)

    def find_attribute_with_most_information_gain(self, dataset):
        # Returns the gain

        best_attribute_gain = 0
        for attr in dataset.header:
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
