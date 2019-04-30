import nltk
import re

"""
Functions generating features
inputs are sentence, i.e. string
outputs are numerical
"""


def frac_of_pos(sentence):
    text = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(text)
    total = float(len(pos_tags))
    noun, verb, prep = 0, 0, 0
    # print (pos_tags)
    for _, pos in pos_tags:
        if pos[0] == "N":
            noun += 1
            continue
        if pos[:2] == "IN":
            prep += 1
            continue
        if pos[0] == "V":
            verb += 1
            continue

    return [noun / total, verb / total, prep / total]


def frac_of_years_and_numbers(sentence):
    words = re.split("\W+", sentence)
    total = float(len(words))
    year, number = 0, 0

    try:
        for w in words:
            if w.isdigit() and int(w) > 1900 and int(w) < 2100:
                year += 1
                number += 1
            elif w.isdigit():
                number += 1

        return [year / total, number / total]
    except:
        return [year / total, number / total]


def cue_words(sentence, cue_words_list=["according", "likely", "example"]):
    result = []
    stemmer = nltk.stem.porter.PorterStemmer()
    cue_words_list = [stemmer.stem(x) for x in cue_words_list]
    words = [stemmer.stem(x) for x in sentence.split()]

    for cw in cue_words_list:
        if cw in words:
            result.append(1)
        else:
            result.append(0)

    return result


# def finding_signals(sentence, signals=["finds", "found", "concludes", "concluded", "confirm", "given"]):
#     stemmer = nltk.stem.porter.PorterStemmer()
#     signals_list = {stemmer.stem(x) for x in signals}
#     words = [stemmer.stem(x) for x in sentence.split()]
#     for word in words:
#         if word in signals_list:
#             return [1]
#     return [0]



if __name__ == '__main__':
    sentence = "This matter comes before the Board of Veterans' Appeals (Board) on appeal from a July 2008 rating decision of the Department of Veterans Affairs (VA) Regional Office (RO) in Montgomery, Alabama."
    print(cue_words(sentence))
    print(frac_of_years_and_numbers(sentence))
    print(frac_of_pos(sentence))
