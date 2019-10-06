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
