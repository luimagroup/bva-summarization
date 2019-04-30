from sklearn.ensemble import RandomForestClassifier
from basic_features import basic_features
from gen_features import *
from vec_features import word2vec
from load import read_files
import numpy as np
import pickle
import os


def combine_features(cases_names):
    types = ['Citation', 'LegalRule', 'LegalPolicy', 'PolicyBasedReasoning',
             'ConclusionOfLaw', 'EvidenceBasedFinding', 'EvidenceBasedReasoning',
             'Evidence', 'Procedure', 'Header']
    type_set = set(types)

    x = []
    y = []
    for case_name in cases_names:
        annotated_sentences = read_files([case_name])
        if not os.path.exists('../annotated_casetext/' + case_name[:-4] + '.pickle'):
            pure_sentences = [annotated_sentences[i][0] for i in range(len(annotated_sentences))]
            embeddings = word2vec(pure_sentences)
            # Save embeddings to speed up
            with open('../annotated_casetext/' + case_name[:-4] + '.pickle', 'wb') as handle:
                pickle.dump(embeddings, handle, protocol=2)
        else:
            with open('../annotated_casetext/' + case_name[:-4] + '.pickle', 'rb') as handle:
                embeddings = pickle.load(handle)

        for i in list(range(len(annotated_sentences))):
            sentence, sent_type = annotated_sentences[i]

            label = 0
            if sent_type == 'EvidenceBasedFinding':
                label = 1
            elif sent_type != 'Evidence':
                continue
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


if __name__ == '__main__':
    cases_names = []
    for case_name in os.listdir('../annotated_casetext'):
        if case_name.endswith('txt'):
            cases_names.append(case_name)
    cases_names = np.array(cases_names)

    # 0 - Fact, 1 - Finding
    x_train, y_train = combine_features(cases_names)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)

    # save the model to disk
    pickle.dump(model, open('classifier_level_two.sav', 'wb'))

    print("---------------------Model Complete-------------------------")
