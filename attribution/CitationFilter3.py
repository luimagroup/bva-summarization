# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import re

"""
Current Rules:
-1. length < 5, remove.
0. First word "See", remove.
1. Add regular expression: [A-Za-z]+ v. [A-Za-z]+ and see percentage
2. Fed. Cir
3. Tokenize. Remove all token that contains number. If C.F.R or U.S.C.A. or "(West" takes 50%, then it's a citation.
"""

ANNOT_FILE_PATH = "../../annotated_casetext/"


# ANNOT_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#                               "vet-data/annotated-cases/notebooks/annotated_cases/")


def build_sentence_dict(filepath):
    res = defaultdict(list)
    sentence_num = 0

    with open(filepath, 'rb') as file:
        # print(filepath)
        cur_type = ""
        cur_str = ""
        for line in file:
            line = line.decode('utf-8', errors='replace')

            if line == "\n":
                continue
            elif line.startswith("#") and line.endswith("#\n"):
                if cur_type is not "":
                    res[cur_type].append(cur_str)
                cur_type = line[1: len(line) - 2]
                # print(cur_type)
                cur_str = ""
                sentence_num += 1
            else:
                cur_str += line

        res[cur_type].append(cur_str)
        # print(res.keys())

    if "CitationSentence" in res:
        global total_citations_num
        total_citations_num += len(res["CitationSentence"])
        # print(total_citations_num)
    else:
        total_citations_num += len(res["Citation"])
        # print(total_citations_num)
    return res, sentence_num


def is_sentence_short(sentence, min_length=6):
    tokens = sentence.split()
    return len(tokens) < min_length
    # if len(tokens) <= 5:
    #     return True
    # else:
    #     return False


def starts_with_see(sentence):
    tokens = sentence.split()
    if tokens[0].lower() == "see" or tokens[0].lower() == "see,":
        return True
    else:
        return False


def contains_regex(sentence):
    regEx1 = re.compile('[A-Za-z]+ v. [A-Za-z]+')
    regEx2 = re.compile('Fed. Cir.')

    if regEx1.search(sentence) is not None or regEx2.search(sentence) is not None:
        return True
    else:
        return False


def is_ctt_percent_high(sentence):
    tokens = sentence.split()
    length = len(tokens)
    regEx = re.compile('\d')
    cnt = 0
    num_cnt = 0

    if "Vet. App." in sentence:
        cnt += 2

    for token in tokens:
        if regEx.search(token) is not None:
            num_cnt += 1
        elif "U.S.C.A." in token or "C.F.R." in token or u"§" in token or "(West" in token or u'\xef\xbf\xbd' in token:
            cnt += 1

    if float(cnt) / (length - num_cnt) >= 0.5 or float(num_cnt + cnt) / length >= 0.5:
        return True
    else:
        return False


def all_cap(sentence):
    sentence = sentence.strip().replace(' ', '')
    # all_upper = True
    for ch in sentence:
        if not (ch.isalpha() and ch.isupper()):
            # all_upper = False
            return False
    # return all_upper
    return True


"""
Total number of Sentences: 4743
Total number of Citations: 760
Total number of Citations FILTERED: 753
Recall Rate: 0.9907894736842106

Total number of FILTERED: 1084
Precision Rate: 0.6946494464944649
"""


def citation_classifier(sentence):
    regEx = re.compile("[Tt]he [A-Za-z ]*appeal is [A-Za-z]+.")

    return (is_sentence_short(sentence) or starts_with_see(sentence) or contains_regex(sentence) or is_ctt_percent_high(sentence)) \
           and len(sentence) > 3 and not all_cap(sentence) and regEx.search(sentence) is None


def citation_classifier_relaxed(sentence):
    """
    For post extraction filtering.
    """
    regEx = re.compile("[Tt]he [A-Za-z ]*appeal is [A-Za-z]+.")

    return (is_sentence_short(sentence, 4) or starts_with_see(sentence) or contains_regex(sentence) or is_ctt_percent_high(sentence)) \
           and len(sentence) > 3 and regEx.search(sentence) is None


def remove_citations(sentence_dict):
    res = defaultdict(list)
    for type in sentence_dict:
        for sentence in sentence_dict[type]:
            if (is_sentence_short(sentence) or starts_with_see(sentence) or contains_regex(sentence) or is_ctt_percent_high(sentence)) \
                    and len(sentence) > 3 and not all_cap(sentence):
                res[type].append("!! " + sentence + "")

                global total_filtered
                total_filtered += 1

                if type != "Citation" and type != "CitationSentence":
                    global false_positives
                    false_positives.append(sentence)

                if type == "Citation" or type == "CitationSentence":
                    global total_citations_filtered
                    total_citations_filtered += 1
            else:
                if type == "Citation" or type == "CitationSentence":
                    global false_negatives
                    false_negatives.append(sentence)
                res[type].append(sentence)

    return res


if __name__ == "__main__":
    total_sentence_num = 0
    total_citations_num = 0
    total_citations_filtered = 0
    total_filtered = 0
    false_negatives = []
    false_positives = []

    for filename in os.listdir(ANNOT_FILE_PATH):
        # print(filename)
        if filename.startswith('.'):
            # print(filename)
            continue
        filepath = ANNOT_FILE_PATH + filename
        sentence_dict, sentence_num = build_sentence_dict(filepath)
        total_sentence_num += sentence_num

        res = remove_citations(sentence_dict)
        with open("results.txt", 'w') as output:
            output.write("========================= " + filename + " =========================\n")
            for type in sentence_dict:
                output.write("############  " + type + "  ##########\n")
                length = len(sentence_dict[type])
                for i in range(length):
                    output.write(str(i) + ":------------------\n")
                    output.write(sentence_dict[type][i])
                    output.write("\n")
                    output.write(res[type][i])
                    output.write("\n")

    print("### False Negatives: ###")
    for sentence in false_negatives:
        print(sentence)

    print()
    print("### False Positives: ###")
    for sentence in false_positives:
        print(sentence)

    print("Total number of Sentences: " + str(total_sentence_num))
    print("Total number of Citations: " + str(total_citations_num))
    print("Total number of Citations FILTERED: " + str(total_citations_filtered))
    print("Recall Rate: " + str(float(total_citations_filtered) / total_citations_num))
    print()
    print("Total number of FILTERED: " + str(total_filtered))
    print("Precision Rate: " + str(float(total_citations_filtered) / total_filtered))
    print()
