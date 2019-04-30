import os
import re
import sys, time
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from post_extraction_filtering import is_rubbish, filter_sentences
import operator

# create stemmer
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))


def cleanData(sentence):
    # sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence)
    # sentence filter(None, re.split("[.!?", setence))
    ret = []
    sentence = stemmer.stem(sentence)
    for word in sentence.split():
        if not word in stopwords:
            ret.append(word)
    return " ".join(ret)

def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0
    vocab = {}
    for word in sentence:
        vocab[word] = 0

    docInOneSentence = ''
    for t in doc:
        docInOneSentence += (t + ' ')
        for word in t.split():
            vocab[word] = 0

    cv = CountVectorizer(vocabulary=list(vocab.keys()))

    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


def get_clean(case):

    data = open(case, 'r')
    lines = data.readlines()
    clean = []
    originalSentenceOf = {}
    sents = []
    for line in lines:
        if line.startswith("round"):
            temp = ""
        elif line.strip() == "":
            sents.append(" ".join(temp.split()))
        else:
            temp += line

    sents = filter_sentences(sents)
    for sent in sents:
        cl = cleanData(sent)
        originalSentenceOf[cl] = sent
        clean.append(cl)

    return clean, originalSentenceOf


def get_clean_from_filtered_sents(sents):

    clean = []
    originalSentenceOf = {}
    for sent in sents:
        cl = cleanData(sent)
        originalSentenceOf[cl] = sent
        clean.append(cl)

    return clean, originalSentenceOf

# calculate Similarity score each sentence with whole documents
def gen_scores(clean):
    setClean = set(clean)
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])
        score = calculateSimilarity(data, list(temp_doc))
        scores[data] = score

    return scores

# calculate MMR and get summarization
def gen_summary(scores, summary_len=5, alpha=0.5):

    summarySet = []
    while summary_len > 0:
        mmr = {}
        for sentence in list(scores.keys()):
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence, summarySet)
        selected = max(iter(mmr.items()), key=operator.itemgetter(1))[0]
        # print(max(mmr.values()))
        summarySet.append(selected)
        summary_len -= 1

    return summarySet


def get_summary_set(case, summary_len=5, alpha=0.5):
    clean, originalSentenceOf = get_clean(case)
    scores = gen_scores(clean)
    summarySet = gen_summary(scores, summary_len, alpha)
    return [originalSentenceOf[sent].lstrip(" ") for sent in summarySet], [originalSentenceOf[sent].lstrip(" ") for sent in clean]


def get_summary_set_from_filtered(sents, summary_len=4, alpha=0.5):
    clean, originalSentenceOf = get_clean_from_filtered_sents(sents)
    if len(sents) < summary_len:
        return sents, clean
    scores = gen_scores(clean)
    summarySet = gen_summary(scores, summary_len, alpha)
    return [originalSentenceOf[sent].lstrip(" ") for sent in summarySet], [originalSentenceOf[sent].lstrip(" ") for sent in clean]
# rint str(time.time() - start)

if __name__ == '__main__':
    
    clean, originalSentenceOf = get_clean(sys.argv[1])
    scores = gen_scores(clean)
    summarySet = gen_summary(scores)

    print('\nSummary:\n')
    for sentence in summarySet:
        print(originalSentenceOf[sentence].lstrip(' '))
    print()

    print('=============================================================')
    print('\nOriginal Passages:\n')
    from termcolor import colored

    for sentence in clean:
        if sentence in summarySet:
            print(colored(originalSentenceOf[sentence].lstrip(' '), 'red'))
        else:
            print(originalSentenceOf[sentence].lstrip(' '))

