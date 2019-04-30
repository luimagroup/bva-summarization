"""
    Post extraction filtering.

    Focus on precision. Only filter rubbishes that is certainly useless to summarization

"""

import os
# import grammar_check
from CitationFilter3 import citation_classifier_relaxed as citation_classifier
from pprint import pprint


# grammar_checker = grammar_check.LanguageTool('en-US')

def remove_dup(seq):
    """
    https://stackoverflow.com/questions/480214/
    how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def filter_sentences(sents):
    """
    :param sents: list of sentences
    :return list of sentences

    Redundant sentences are removed, while order is preserved.

    """
    return remove_dup(filter(lambda x: not is_rubbish(x), sents))


def is_rubbish(sent):
    """
    :param sent: sentence represented as string
    :return boolean value indicating if it is a useless for summarization
    """
    criteria = [
        lambda x: not x,
        lambda x: x == x.upper(),
        lambda x: citation_classifier(x) and not is_long(x),
        # lambda x: len(grammar_checker.check(x)) > 0, # potentially unsafe, need to pay attention
        lambda x: too_many_capital_starts(x, 0.5), # potentially unsafe, need to pay attention
    ]

    return any(criterion(sent) for criterion in criteria)

def is_long(x, thresh=30, lower_case_thresh=10):
    tokens = x.split()
    lower_case_tokens = list(filter(lambda x: x.islower(), tokens))
    return len(tokens) > thresh or len(lower_case_tokens) > lower_case_thresh

def too_many_capital_starts(x, thresh=0.74):
    words = [w.strip() for w in x.split()]
    cnt = sum([1 if w and w[0].isupper() else 0 for w in words])
    return float(cnt) / len(words) > thresh


######## testing functions ########
def gen_sents_from_log(case):
    base_dir = "./log/per_case/"
    casefile = os.path.join(base_dir, case)
    f = open(casefile, "r")
    lines = f.readlines()
    sents = []
    sen = ""
    for line in lines:
        if not line.startswith("round "):
            sen += line
        else:
            sents.append(" ".join(sen.split()))
            sen = ""

    return sents


def filtered_sents(sents):
    return list(filter(is_rubbish, sents))


def myprint(lst):
    for x in lst:
        print(x)


# if __name__ == '__main__':
#     # sent = "The notification complied with the requirements of Quartuccio v. Principi, 16 Vet. App. 183 (2002), identifying the evidence necessary to substantiate a claim and the relative duties of VA and the claimant to obtain evidence."
#     # print(citation_classifier(sent))
#     # print(is_rubbish("ORDER"))
#     base_dir = "./log/per_case/"
#     for case in os.listdir(base_dir):
#         sents = gen_sents_from_log(case)
#         print(case)
#         print("============ filtered sentences ============")
#         myprint(filtered_sents(sents))
#         print()
#         print("============= filtering result =============")
#         myprint(filter_sentences(sents))
#         print()
#         pause = input("press enter to continue")



