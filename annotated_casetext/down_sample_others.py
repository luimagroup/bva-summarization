import re
import codecs
import os
import random


def read_files(cases_names, documents_dir='./'):
    """
     Read annotated case files
    :param documents_dir: directory of case files
    :return: each annotated sentence and its type
    """
    annotated_lines = []
    line_types = set()
    type_pattern = re.compile("#[\s\S]*#")
    for case_name in cases_names:
        with codecs.open(os.path.join(documents_dir, case_name.strip()), 'rb', encoding='utf-8') as case:
            lines = []
            line_type = ''
            for line in case:
                # line = line.decode('utf-8', errors='replace').strip()
                line = line.strip()
                # If starts to meet a new line
                if line == '':
                    if len(lines) != 0:
                        annotated_line = ' '.join(lines)
                        annotated_lines.append((annotated_line, line_type))
                    lines = []
                    line_type = ''
                elif re.match(type_pattern, line):
                    line_type = line.strip('#')
                    line_types.add(line_type)
                else:
                    lines.append(line)

    print(line_types)
    return annotated_lines


if __name__  == '__main__':
    others_set = {'Citation', 'Sentence', 'Header', 'LegalPolicy', 'LegalRule', 'CitationSentence', 'ConclusionOfLaw',
                  'LegalRuleSentence', 'Procedure'}

    cases = []
    for case_name in os.listdir('./'):
        case_name = case_name.strip()
        if not case_name.endswith('.txt'):
            continue
        cases.append(case_name)

    annotated_lines = read_files(cases)
    other_sents = set()
    care_sents = set()
    for line in annotated_lines:
        if line[1] in others_set:
            other_sents.add(line[0])
        else:
            care_sents.add(line[0])

    print("Finding % Fact sentences num:")
    print(len(care_sents))
    print("Other sentences num:")
    print(len(other_sents))

    # Downsample some other sentences from all the other type sentences
    down_sample = random.sample(list(other_sents), 500)
    other_train = set(random.sample(down_sample, 375))
    down_sample = set(down_sample)
    other_val = down_sample - other_train

    with codecs.open("sampled_other_train.txt", 'w', encoding='utf-8') as f:
        for sent in other_train:
            f.write(sent + "\n")

    with codecs.open("sampled_other_val.txt", 'w', encoding='utf-8') as f:
        for sent in other_val:
            f.write(sent + "\n")
