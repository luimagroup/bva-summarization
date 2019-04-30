import numpy as np
import json
import keras
import os
import pickle
from IntegratedGradients import integrated_gradients
from sent_seg import segmented_sents, filter_sentences
from pprint import pprint


def generate_testcases(path, limit=50):
    i = 1
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'rb') as file:
            x, y = pickle.load(file)
            yield (np.expand_dims(x, axis=-1), filename.split('.')[0], y)  # X, caseid, outcome
            i += 1
            if i > limit:
                break


def get_top_sentence(model_path, test_gen, k=5, include_negative=False):

    """
    :param modelpath: the model saved
    :param test_gen: a generator of test data
    :param k: select top k sentences
    :param include_negative: also inlcude negative sentences
    :return: generator of dict with attibutes of 'caseid', 'outcome', 'positive', (optional) 'negative'
    """
    model = keras.models.load_model(model_path)
    ig = integrated_gradients(model)
    # test_gen = generate_testcases(test_data_path)
    # out = [-1: "denied", 0: "remanded", 1: "granted"]

    for X, caseid, outcome in test_gen:
        sent_strings = filter_sentences(segmented_sents(caseid))
        attributions = np.sum(ig.explain(X, num_steps=50, outc=outcome+1), axis=1).tolist()
        attributions = [(i, v) for i, v in enumerate(attributions)]
        positives = sorted(attributions, key=lambda x: x[1], reverse=True)
        negatives = sorted(attributions, key=lambda x: x[1])

        result = {"positive":[], "caseid": caseid, "outcome": outcome}
        # print("###### %s %d ########" % (caseid, outcome))
        # print("Top 5 Positive Sentences\n")
        for i, v in positives[:k]:
            result["positive"].append((i, v, sent_strings[i]))
        # print("Top 5 Negative Sentences\n")
        if include_negative:
            result["negative"] = []
            for i, v in negatives[:k]:
                result["negative"].append((i, v, sent_strings[i]))

        yield result


if __name__ == '__main__':
    test_data_path = "../sentence_encoder/tempTrain/"
    top_sents = get_top_sentence('weights.best.hdf5', generate_testcases(test_data_path), 5)
    for i, ts in enumerate(top_sents):
        if i > 10:
            break
        pprint(ts)

