"""
python type_classifier/experiment.py
"""
from sklearn.ensemble import RandomForestClassifier
from basic_features import basic_features
from gen_features import *
from vec_features import word2vec
from load import read_files
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
import os
import pickle
import codecs


finding_types = {'Reasoning'}
fact_types = {'Evidential Support'}
annotated_dir = '../../new_annotated_casetext/'
downsample_dir = '../../annotated_casetext/'


def combine_features(cases_names):
    x = []
    y = []
    for case_name in cases_names:
        annotated_sentences = read_files([case_name], annotated_dir)
        if not os.path.exists(annotated_dir + case_name[:-4] + '.pickle'):
            pure_sentences = [annotated_sentences[i][0] for i in range(len(annotated_sentences))]
            try:
                embeddings = word2vec(pure_sentences)
            except:
                print("USC failed", case_name)
                continue
            # Save embeddings to speed up
            with open(annotated_dir + case_name[:-4] + '.pickle', 'wb') as handle:
                pickle.dump(embeddings, handle, protocol=2)
        else:
            with open(annotated_dir + case_name[:-4] + '.pickle', 'rb') as handle:
                embeddings = pickle.load(handle)

        for i in range(len(annotated_sentences)):
            sentence, sent_type = annotated_sentences[i]

            if sent_type in finding_types or sent_type in fact_types:
                label = 1
            # elif sent_type in fact_types:
            #     label = 2
            else:
                label = 0
            y.append(label)

            # Word2Vec feature
            feature1 = embeddings[i].tolist()
            # Basic features: sentence length, number of periods, percent of characters that are capitalized
            feature2 = basic_features(sentence)
            # Fraction of POS feature
            feature3 = frac_of_pos(sentence)
            # Fraction of years and numbers feature
            feature4 = frac_of_years_and_numbers(sentence)
            # Cue words feature
            feature5 = cue_words(sentence)

            features = feature1 + feature2 + feature3 + feature4 + feature5
            x.append(np.array(features))

    x = np.array(x)
    y = np.array(y)
    return x, y


def combine_features_negative_downsample(cases_names, downsample_file):
    x = []
    y = []
    for case_name in cases_names:
        annotated_sentences = read_files([case_name], annotated_dir)
        if not os.path.exists(annotated_dir + case_name[:-4] + '.pickle'):
            pure_sentences = [annotated_sentences[i][0] for i in range(len(annotated_sentences))]
            try:
                embeddings = word2vec(pure_sentences)
            except:
                print("USC failed", case_name)
                continue
            # Save embeddings to speed up
            with open(annotated_dir + case_name[:-4] + '.pickle', 'wb') as handle:
                pickle.dump(embeddings, handle, protocol=2)
        else:
            with open(annotated_dir + case_name[:-4] + '.pickle', 'rb') as handle:
                embeddings = pickle.load(handle)

        for i in range(len(annotated_sentences)):
            sentence, sent_type = annotated_sentences[i]

            if sent_type in finding_types or sent_type in fact_types:
                label = 1
            # elif sent_type in fact_types:
            #     label = 2
            else:
                label = 0
            y.append(label)

            # Word2Vec feature
            feature1 = embeddings[i].tolist()
            # Basic features: sentence length, number of periods, percent of characters that are capitalized
            feature2 = basic_features(sentence)
            # Fraction of POS feature
            feature3 = frac_of_pos(sentence)
            # Fraction of years and numbers feature
            feature4 = frac_of_years_and_numbers(sentence)
            # Cue words feature
            feature5 = cue_words(sentence)

            features = feature1 + feature2 + feature3 + feature4 + feature5
            x.append(np.array(features))

    # For downsampled other type sentences
    annotated_sentences = []
    with codecs.open(downsample_dir + downsample_file, 'r', encoding='utf-8') as f:
        for other_sent in f.readlines():
            other_sent = other_sent.strip()
            if len(other_sent) == 0:
                continue
            annotated_sentences.append(other_sent)

    if not os.path.exists(downsample_dir + downsample_file[:-4] + '.pickle'):
        embeddings = word2vec(annotated_sentences)
        # Save embeddings to speed up
        with open(downsample_dir + downsample_file[:-4] + '.pickle', 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=2)
    else:
        with open(downsample_dir + downsample_file[:-4] + '.pickle', 'rb') as handle:
            embeddings = pickle.load(handle)

    for i in range(len(annotated_sentences)):
        sentence = annotated_sentences[i]
        y.append(0)
        # Word2Vec feature
        feature1 = embeddings[i].tolist()
        # Basic features: sentence length, number of periods, percent of characters that are capitalized
        feature2 = basic_features(sentence)
        # Fraction of POS feature
        feature3 = frac_of_pos(sentence)
        # Fraction of years and numbers feature
        feature4 = frac_of_years_and_numbers(sentence)
        # Cue words feature
        feature5 = cue_words(sentence)

        features = feature1 + feature2 + feature3 + feature4 + feature5
        x.append(np.array(features))

    x = np.array(x)
    y = np.array(y)
    return x, y


# For using all new annotated cases & downsampled other type sentences from old annotated data
if __name__ == '__main__':
    train_cases = []
    val_cases = []
    with open(annotated_dir + 'train_cases.txt', 'r') as f:
        for case_name in f.readlines():
            case_name = case_name.strip()
            if not case_name.endswith(".txt"):
                continue
            train_cases.append('annot_' + case_name)

    with open(annotated_dir + 'validation_cases.txt', 'r') as f:
        for case_name in f.readlines():
            case_name = case_name.strip()
            if not case_name.endswith(".txt"):
                continue
            val_cases.append('annot_' + case_name)

    target_names = ['Others', 'Finding & Fact']
    total_prediction = []
    total_gold = []

    print("TRAIN CASES:", train_cases)
    print("VAL CASES:", val_cases)

    x_train, y_train = combine_features_negative_downsample(train_cases, 'sampled_other_train.txt')
    x_test, y_test = combine_features_negative_downsample(val_cases, 'sampled_other_val.txt')

    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    total_prediction.extend(prediction)
    total_gold.extend(y_test)
    print("---------------------OVERALL STATS-------------------------")
    print(classification_report(y_test, prediction, target_names=target_names))


# For using only new annotated 92 cases
# if __name__ == '__main__':
#     train_cases = []
#     val_cases = []
#     with open(annotated_dir + 'train_cases.txt', 'r') as f:
#         for case_name in f.readlines():
#             case_name = case_name.strip()
#             if not case_name.endswith(".txt"):
#                 continue
#             train_cases.append('annot_' + case_name)
#
#     with open(annotated_dir + 'validation_cases.txt', 'r') as f:
#         for case_name in f.readlines():
#             case_name = case_name.strip()
#             if not case_name.endswith(".txt"):
#                 continue
#             val_cases.append('annot_' + case_name)
#
#     target_names = ['Others', 'Finding & Fact']
#     total_prediction = []
#     total_gold = []
#
#     print("TRAIN CASES:", train_cases)
#     print("VAL CASES:", val_cases)
#
#     x_train, y_train = combine_features(train_cases)
#     x_test, y_test = combine_features(val_cases)
#
#     model = RandomForestClassifier(n_estimators=100)
#     model.fit(x_train, y_train)
#     prediction = model.predict(x_test)
#     total_prediction.extend(prediction)
#     total_gold.extend(y_test)
#     print("---------------------OVERALL STATS-------------------------")
#     print(classification_report(y_test, prediction, target_names=target_names))
