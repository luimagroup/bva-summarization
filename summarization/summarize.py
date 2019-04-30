import os, sys
from post_extraction_filtering import filter_sentences
from template_sents_extraction import find_appeal, find_issue, find_conclusion, find_service_time
from mmr import get_summary_set_from_filtered
from type_classifier.annotate import annotate_sentences
import pickle
import codecs
import numpy as np


BASE_DIR = "./sample_case_masked_sents_60/"
OUTPUT_DIR = "./sample_case_summaries_60/"


def summarize(case, summary_len, verbose=1):
    """
    Produce a 7-sentence summarization for a given case.
    :param case: case file name
    :param summary_len: summary length
    :param verbose: verbose level
    :return: list of sentences
    """

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
    middle_set, middle_clean = get_summary_set_from_filtered(middle, summary_len=middle_len, alpha=0.5)
    # MMR on fact sentences
    # fact_set, fact_clean = get_summary_set_from_filtered(fact, summary_len=middle_len-len(finding_set))

    # MMR on other sentences
    other_set, other_clean = get_summary_set_from_filtered(sents, summary_len=middle_len-len(middle_set), alpha=0.5)
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
    print_summary(case, summary_set, clean, verbose)
    return summary_set, discard_len, total_len


def print_summary(case, summary_set, clean, verbose=1):

    if not verbose:
        return

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    print('\nSummary:\n')
    # with codecs.open('./summaries_7/' + case, 'w', encoding='utf-8') as f:
    with codecs.open(os.path.join(OUTPUT_DIR, case), 'w', encoding='utf-8') as f:
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


def gold_in_masked_sents():
    ref_folder = "ordered_ref_summaries/"
    masked_sents_folder = "sample_case_masked_sents_60/"
    ref_cases = os.listdir(ref_folder)
    masked_cases = os.listdir(masked_sents_folder)
    ref_total = 0.
    matched_total = 0.
    num_cases = 0.
    zero_match_cases = 0

    for ref_case in ref_cases:
        count = 0
        if ref_case not in masked_cases:
            continue
        num_cases += 1
        print(ref_case)
        with codecs.open(ref_folder+ref_case, 'r', encoding='utf-8', errors='ignore') as file:
            ref_summary = [sent.strip() for sent in file.readlines()[3:-1]]
            ref_total += len(ref_summary)
            print(len(ref_summary))
        with codecs.open(masked_sents_folder+ref_case, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            masked_sents = []
            temp = ""
            for line in lines:
                line = line.strip()
                if line.startswith("round"):
                    temp = ""
                elif line.strip() == "":
                    masked_sents.append(temp)
                else:
                    temp += line
            masked_sents.append(temp)
            masked_sents = set(masked_sents)
            print(len(masked_sents))
        for sent in ref_summary:
            for masked_sent in masked_sents:
                if fuzzy_match(sent, masked_sent):
                    print(sent)
                    count += 1
                    matched_total += 1
        print(count)
        if count == 0:
            zero_match_cases += 1
        print()
    print(num_cases)
    print(matched_total/ref_total)
    print(matched_total/num_cases)
    print(zero_match_cases)
    print(ref_total/num_cases)


def fuzzy_match(sent1, sent2):
    return sent1.lower() in sent2.lower()


if __name__ == '__main__':
    # gold_in_masked_sents()

    reload(sys)
    sys.setdefaultencoding('utf8')
    test_file = '../new_annotated_casetext/validation_cases.txt'
    case_names = []
    # discard_lens_file = open("discard_lens.txt", "a")
    # discard_lens = []
    with open(test_file, 'r') as f:
        for case_name in f.readlines():
            case_name = case_name.strip()
            if not case_name.endswith(".txt"):
                continue
            case_names.append(case_name)

    with open('gold_summaries_len_map.pickle', 'rb') as handle:
        gold_summaries_len = pickle.load(handle)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for case_name in case_names:
        # if os.path.exists(OUTPUT_DIR + case_name):
        #     continue
        if case_name[:-4] not in gold_summaries_len:
            continue
        print(case_name)
        try:
            target_len = gold_summaries_len[case_name[:-4]]
            _, discard_len, total_len = summarize(case_name, target_len, verbose=1)
            # discard_lens_file.write(case_name + "\t" + str(total_len) + "\t" + str(discard_len) + "\n")
            # discard_lens.append(discard_len)
        except Exception as e:
            print(e)
            print(case_name, "failed")

    # discard_lens_file.close()
    # print(np.mean(discard_lens))
