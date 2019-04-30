import os, sys
from post_extraction_filtering import filter_sentences
from template_sents_extraction import find_appeal, find_issue, find_conclusion, find_service_time
from mmr import get_summary_set_from_filtered
from type_classifier.annotate import annotate_sentences
import pickle
import codecs
import numpy as np


BASE_DIR = "./sample_case_masked_sents_60/"
OUTPUT_DIR = "./sample_case_summaries_60_test/"


def summarize(case, summary_len, target_dir, verbose=1):
    if ".txt" not in case:
        case += ".txt"

    # Read the file and get masked sentences
    with codecs.open(BASE_DIR + case, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sents = []
    join_line = []
    indices = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('round ') and line.endswith(":"):
            indices.append(int(line[line.rfind(" ") + 1:-1]))
            sent = ' '.join(join_line)
            if len(sent) > 0:
                join_line = []
                sents.append(sent)
        else:
            join_line.append(line)

    sent = ' '.join(join_line)
    if len(sent) > 0:
        sents.append(sent)

    sent2index = {k: indices[v] for v, k in enumerate(sents)}

    sents = filter_sentences(sents)
    appeal = find_appeal(case) # Alternatively, could read from pre-extracted file
    issue = find_issue(case) # Alternatively, could read from pre-extracted file
    service_time = find_service_time(case)
    conclusion = find_conclusion(case, sents)
    sents = [x for x in sents if x != conclusion]

    ##### THIS PART NEED TO CHANGE, NOW A NAIVE METHOD #####
    with open('./type_classifier/random_forest_classifier.sav', 'rb') as handle:
        model = pickle.load(handle)
    # fact, finding = annotate_sentences(model, sents)
    # for sent in fact:
    #     sents.remove(sent)
    # for sent in finding:
    #     sents.remove(sent)
    total_len = len(sents)
    middle = annotate_sentences(model, sents)
    discard_len = total_len - len(middle)
    for sent in middle:
        sents.remove(sent)

    summary_set = []
    clean = []
    if service_time is None:
        middle_len = summary_len - 3    # Number of fact and finding sentences
    else:
        middle_len = summary_len - 4
    # MMR on finding & fact sentences
    middle_set, middle_clean = get_summary_set_from_filtered(middle, summary_len=middle_len, alpha=0.8)
    # MMR on fact sentences
    # fact_set, fact_clean = get_summary_set_from_filtered(fact, summary_len=middle_len-len(finding_set))

    # MMR on other sentences
    other_set, other_clean = get_summary_set_from_filtered(sents, summary_len=middle_len-len(middle_set), alpha=0.8)
    summary_set.extend(other_set)
    clean.extend(other_clean)
    summary_set.extend(middle_set)
    clean.extend(middle_clean)
    # summary_set.extend(finding_set)
    # clean.extend(finding_clean)

    ########################################################
    # Sort finding/fact section by original index.
    summary_set.sort(key=lambda s: sent2index[s])

    if service_time is None:
        summary_set = [appeal, issue] + summary_set + [conclusion]
    else:
        summary_set = [appeal, issue, service_time] + summary_set + [conclusion]
    print_summary(case, summary_set, clean, target_dir, verbose)
    return summary_set, discard_len, total_len


def print_summary(case, summary_set, clean, target_dir, verbose=1):
    if not verbose:
        return

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    print('\nSummary:\n')
    # with codecs.open('./summaries_7/' + case, 'w', encoding='utf-8') as f:
    with codecs.open(os.path.join(target_dir, case), 'w', encoding='utf-8') as f:
        for sentence in summary_set:
            f.write(sentence + '\n')
            print(sentence.lstrip(' '))

    if verbose == 2:
        print('=============================================================')
        print('\nOriginal Passages:\n')
        from termcolor import colored

        for sentence in clean:
            if sentence in summary_set:
                print(colored(sentence.lstrip(' '), 'red'))
            else:
                print(sentence.lstrip(' '))


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    case_names = "0704797.txt 0204881.txt 0206985.txt 0218389.txt 0207638.txt 1339942.txt 0825614.txt 0924745.txt " \
                 "1454469.txt 1631247.txt 0403957.txt 0121258.txt 1727981.txt 1445242.txt 1203607.txt 0017145.txt " \
                 "9932395.txt 0206017.txt 0010438.txt 1640466.txt".split()

    with open('gold_summaries_len_map_per_annotator.pickle', 'rb') as handle:
        gold_summaries_len = pickle.load(handle)

    for each_annotator in gold_summaries_len:
        # if each_annotator != '58a0bf4f8424bd4f65e2be57':
        #     continue
        target_path = os.path.join(OUTPUT_DIR, each_annotator)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for case_name in case_names:
            # if os.path.exists(os.path.join(target_path, case_name)):
            #     continue
            if case_name[:-4] not in gold_summaries_len[each_annotator]:
                continue
            print(case_name)
            try:
                target_len = gold_summaries_len[each_annotator][case_name[:-4]]
                _, discard_len, total_len = summarize(case_name, target_len, target_path, verbose=1)
            except Exception as e:
                print(e)
                print(case_name, "failed")
