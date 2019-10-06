from random import choice
import utils


class CsvDataset:
    n_folds = 10
    def __init__(self, filename=None):
        self.filename = filename
        if filename == None:
            self.header = []
            self.items = []

        else:
            with open(filename) as csvfile:
                lines = csvfile.readlines()

            header = lines[0].split(',')
            # Removes double quotes, \n and transforms in lowercase
            self.header = [x.replace('"', '').replace('\n', '').lower() for x in header]

            self.items = []
            for one_line in lines[1::]:
                one_item = self._line_to_dict(one_line)
                self.items.append(one_item)

            assert len(self.items) == len(lines) - 1  # Asserts everything was added to itens

    def _line_to_dict(self, line):
        line = line.split(',')
        one_item = {}
        for x, y in zip(line, self.header):
            x = x.replace("'", '').replace('\n', '')
            try:
                one_item[y] = float(x)
                utils.log(f"Adding numeric item to {y}")
            except ValueError:
                one_item[y] = x
                utils.log(f"Adding categorical item to {y}")

        assert len(one_item) > 1  # Avoid empty

        return one_item

    def filter_dataset_categorical(self, attr, c):
        new_dataset = CsvDataset()
        new_dataset.header = self.header.copy()
        new_dataset.items = [x for x in self.items if x[attr] == c]

        return new_dataset

    def filter_dataset_numerical(self, attr, div):
        smaller_dataset = CsvDataset()
        smaller_dataset.header = self.header.copy()
        smaller_dataset.items = [x for x in self.items if x[attr] < div]

        bigger_dataset = CsvDataset()
        bigger_dataset.header = self.header.copy()
        bigger_dataset.items = [x for x in self.items if x[attr] > div]

        return smaller_dataset, bigger_dataset

    def filter_attr(self, attrs):
        new_dataset = CsvDataset()
        new_dataset.header = attrs.copy()
        new_dataset.items = [{x: i[x] for x in attrs} for i in self.items]

        return new_dataset

    def bootstrap(self):
        new_dataset = CsvDataset()
        new_dataset.header = self.header
        new_dataset.items = [choice(self.items) for _ in range(len(self.items))]

        return new_dataset

    def get_folds(self, n):
        # n is the index of the val fold
        classes_item = {}
        for i in self.items:
            if i['class'] not in classes_item.keys():
                classes_item[i['class']] = [i]
            else:
                classes_item[i['class']].append(i)

        classes_item = {x: utils.chunk_it(classes_item[x], self.n_folds) for x in classes_item.keys()}

        train_data = []
        test_data = []

        for k in classes_item.keys():
            for i in range(self.n_folds):
                if i == n:
                    train_data += classes_item[k][i]
                else:
                    test_data += classes_item[k][i]

        train_dataset = CsvDataset()
        train_dataset.header = self.header
        train_dataset.items = train_data

        test_dataset = CsvDataset()
        test_dataset.header = self.header
        test_dataset.items = test_data

        return train_dataset, test_dataset

    def __iter__(self):
        return self.items.__iter__()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

