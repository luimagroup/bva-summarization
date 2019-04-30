import os, sys
import codecs
import matplotlib.pyplot as plt
import re


def read_file(case_path):
    # Read the file and get masked sentences
    with codecs.open(case_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    indices = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if (line.startswith('round ') or line.startswith('top ')) and 'index' in line:
            indices.append(int(line[line.rfind(" ") + 1:-1]))

    return set(indices)


def compare_files(file1, file2):
    indices_set_1 = read_file(file1)
    indices_set_2 = read_file(file2)
    return len(indices_set_1.intersection(indices_set_2))


def compare(dir1, dir2):
    dir1_files = os.listdir(dir1)
    results = []
    for case_name in dir1_files:
        if not case_name.endswith('.txt'):
            continue
        results.append(compare_files(os.path.join(dir1, case_name), os.path.join(dir2, case_name)))

    x = list(range(0, 41))
    y = [0 for i in range(41)]
    for num in results:
        y[num] += 1
    plt.bar(x, y, color="blue")
    plt.title("Histogram of Number of Different Sentences between Top 40 Attributed Sentences and Masked Sentences from 40 Rounds")
    # plt.show()
    plt.xlabel("number of overlapped sentences")
    plt.ylabel("number of cases")
    plt.savefig('diff.png')


if __name__ == '__main__':
    all_rounds_dir = sys.argv[1]
    first_round_dir = sys.argv[2]
    compare(all_rounds_dir, first_round_dir)
