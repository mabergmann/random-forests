from utils import log


class CsvDataset:
    def __init__(self, filename):
        with open(filename) as csvfile:
            lines = csvfile.readlines()

        header = lines[0].split(',')
        # Removes double quotes, \n and transforms in lowercase
        self.header = [x.replace('"', '').replace('\n', '').lower() for x in header]

        self.itens = []
        for one_line in lines[1::]:
            one_item = self._line_to_dict(one_line)
            self.itens.append(one_item)

        assert len(self.itens) == len(lines) - 1  # Asserts everything was added to itens

    def _line_to_dict(self, line):
        line = line.split(',')
        one_item = {}
        for x, y in zip(line, self.header):
            try:
                one_item[y] = float(x)
                log(f"Adding numeric item to {y}")
            except ValueError:
                one_item[y] = x
                log(f"Adding categorical item to {y}")

        assert len(one_item) > 1  # Avoid empty line

        return one_item
