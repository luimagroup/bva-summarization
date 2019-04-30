'''
get paragraph segments from sentence segments
'''

import pickle
from pprint import pprint
import re
from CitationFilter3 import citation_classifier, all_cap
'''
return:
sent2para: inverted index array: [P0, P0, P1, P1, P2 .... P100 P100]
para_sent_pairs: list of list of tuples: para sent start/end pair
'''


def filter_sentences(sents):
    after_sents = []

    pattern = re.compile("^[0-9]+.$")
    for sentence in sents:
        if citation_classifier(sentence) or all_cap(sentence) or re.match(pattern, sentence):
            continue
        for ch in [u"_", "\r", "\n", "CONCLUSION OF LAW", "REASONS AND BASES FOR FINDINGS AND CONCLUSION", "ORDER",
                   "FINDINGS OF FACT", "INTRODUCTION", "THE ISSUE", "REPRESENTATION", "WITNESS AT HEARING ON APPEAL",
                   "ATTORNEY FOR THE BOARD"]:
            sentence = sentence.replace(ch, "")
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        after_sents.append(sentence)

    return after_sents


def segmented_sents(caseid):

    name = caseid + '.pkl'
    # TODO: Change to relative path
    # path = '../bva-segments/'
    path = '../../bva-segments/'

    file = open(path+name, 'rb')
    sent_list = pickle.load(file)

    with open('../../all-bva-decisions/' + caseid + '.txt', 'rb') as file:
        case_str = file.read().decode("ISO8859-1")

    para_i = 0
    sent2para = []
    AFTER_INTRO = False

    # list of list of tuples: para sent start/end pair
    para_sent_pairs = []
    sent_pairs = []

    # sent string list
    sent_strings = []

    # para2idx
    para2idx = {}
    idx2para = {}
    para_string = []

    for item in sent_list:
        # print(item)
        sent_string = (case_str[item[0]:item[1]])
        # print(sent_string)
        ending = case_str[item[1] - 4:item[1] + 16]  # hardcoded to be most robust

        if AFTER_INTRO:
            sent2para.append(para_i)
            sent_strings.append(sent_string)

            # check ending and see if it's end of paragraph
            if ending.count('\r\n') >= 2:
                para_sent_pairs.append(sent_pairs)
                para = '  '.join(para_string)
                para2idx[para[:50]] = para_i
                idx2para[para_i] = para[:50]
                para_i += 1

                sent_pairs = []

        if 'INTRODUCTION' in sent_string:
            AFTER_INTRO = True

    return sent_strings


if __name__ == '__main__':
    print(segmented_sents('1434436'))