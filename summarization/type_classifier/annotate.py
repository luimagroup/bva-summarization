import pickle
# from summarization.type_classifer.basic_features import basic_features
# from summarization.type_classifer.gen_features import *
# from summarization.type_classifer.vec_features import word2vec
from basic_features import basic_features
from gen_features import *
from vec_features import word2vec
import numpy as np
import sys
import codecs
import nltk


def finding_by_heuristics(sentence):
    signals = ["finds", "found", "conclude", "concluded", "confirm", "given"]
    stemmer = nltk.stem.porter.PorterStemmer()
    signals_list = {stemmer.stem(x) for x in signals}
    words = [stemmer.stem(x) for x in sentence.split()]
    for word in words:
        if word in signals_list:
            return True
    return False


def annotate_sentences(model, sents):
    cared_sents = []

    embeddings = word2vec(sents)              # If use USC
    for i in range(len(sents)):
        sentence = sents[i]
        if finding_by_heuristics(sentence):
            cared_sents.append(sentence)
            continue
        # Word2Vec feature
        feature1 = embeddings[i].tolist()     # If use USC
        # feature1 = word2vec(sentence)        # Use law2vec
        # Basic features: sentence length, number of periods, percent of characters that are capitalized
        feature2 = basic_features(sentence)
        # Fraction of POS feature
        feature3 = frac_of_pos(sentence)
        # Fraction of years and numbers feature
        feature4 = frac_of_years_and_numbers(sentence)
        # Cue words feature
        feature5 = cue_words(sentence)

        features = np.array(feature1 + feature2 + feature3 + feature4 + feature5).reshape(1, -1)
        type = model.predict(features)
        if type[0] == 1:
            cared_sents.append(sentence)

    return cared_sents


if __name__ == '__main__':
    case_name = sys.argv[1]
    with codecs.open(case_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    join_line = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('round'):
            sent = ' '.join(join_line)
            if len(sent) > 0:
                join_line = []
                sentences.append(sent)
        else:
            join_line.append(line)

    sent = ' '.join(join_line)
    if len(sent) > 0:
        sentences.append(sent)

    with open('random_forest_classifier.sav', 'rb') as handle:
        model = pickle.load(handle)
    cared = annotate_sentences(model, sentences)
    print("================== Fact & Finding Sentences ==================")
    print(cared)
