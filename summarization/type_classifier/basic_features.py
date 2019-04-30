# coding: utf-8

import nltk


def basic_features(sentence):
    '''
    Input: raw sentence string
    Output:
    sentence length (# words): int
    number of periods: int
    percent of characters that are capitalized: float
    '''
    # sentence length
    sentLen = len(nltk.word_tokenize(sentence))

    numPeriods = sentence.count('.')

    # percent of characters
    numCapChars = numChars = 0
    for char in sentence:
        if char.isupper():
            numCapChars += 1
        if char.isalpha():
            numChars += 1
    if numChars == 0:
        percentCapChars = 0.0
    else:
        percentCapChars = float(numCapChars) / numChars

    return [sentLen, numPeriods, percentCapChars]


def main():
    print(basic_features("Entitlement to service connection for residuals of a left eye injury."))
    print(basic_features("38 U.S.C.A. § 5107(a)(West 1991)."))


if __name__ == "__main__":
    main()






