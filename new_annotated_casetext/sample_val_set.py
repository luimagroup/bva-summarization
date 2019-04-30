import pickle
import random


if __name__ == '__main__':
    with open('annotator_case_map.pickle', 'rb') as handle:
        annotator_case_map = pickle.load(handle)

    train_cases = []
    val_cases = []
    for annotator in annotator_case_map:
        for case_type in annotator_case_map[annotator]:
            cur_cases = set(annotator_case_map[annotator][case_type])
            if case_type == -1:
                cur_val_cases = set(random.sample(cur_cases, 2))
            elif case_type == 0:
                cur_val_cases = set(random.sample(cur_cases, 3))
            else:
                cur_val_cases = set(random.sample(cur_cases, 1))
            val_cases.extend(list(cur_val_cases))
            train_cases.extend(list(cur_cases - cur_val_cases))

    train_cases = set(train_cases)
    val_cases = set(val_cases)

    intersect = train_cases.intersection(val_cases)
    assert len(intersect) == 0

    train_set_file = 'train_cases.txt'
    with open(train_set_file, 'w') as f:
        for case in train_cases:
            f.write(case + '.txt' + '\n')

    val_set_file = 'validation_cases.txt'
    with open(val_set_file, 'w') as f:
        for case in val_cases:
            f.write(case + '.txt' + '\n')
