import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
# from sentence_encoder import sent_seg
import pickle
import os, sys
from CitationFilter3 import citation_classifier_relaxed, citation_classifier, all_cap
import re

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


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


def encoder(sents, session):
    # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)

  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(sents))
  return np.array(message_embeddings)


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


# Read case_id's and outcomes from file
# and get sentence embeddings
option = sys.argv[1]

f = open("PTSD_%sCases_outcome.txt" % option)
# f = open("sample_case_outcome.txt")
lines = f.readlines()


def get_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


chunks = get_chunks(lines, 100)
input_sentences = tf.placeholder(tf.string, shape=[None])
embedding = embed(input_sentences)

session = tf.InteractiveSession()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
for chunk in chunks:
    all_sentences, outcomes, cases = [], [], []
    for line in chunk:
        casefile, outcome = line.strip().split()
        caseid = casefile[:-4]
        outcome = int(outcome)

        if os.path.isfile('./%sEmbeddings/%s.pickle' % (option, caseid)):
            print(caseid + " already exists, skipped")
            continue

        try:
            seg_sents = segmented_sents(caseid)
            # Filter out garbage sentences in the first place
            filtered_sents = filter_sentences(seg_sents)
            all_sentences.append(filtered_sents)
            outcomes.append(outcome)
            cases.append(caseid)
            print("preprocess " + caseid)
        except Exception as e:
            print(caseid + " cannot be segmented")

    print("preprocess done.")

    for i in range(len(cases)):
        caseid = cases[i]
        outcome = outcomes[i]
        sentences = all_sentences[i]
        try:
            embeddings = embedding.eval(feed_dict={input_sentences: sentences})
            # embeddings = encoder(sentences, session)
            with open('./%sEmbeddings/%s.pickle' % (option, caseid), 'wb') as handle:
                pickle.dump([embeddings, outcome], handle, protocol=2)
            print(caseid)
        except Exception as e:
            print(caseid + " failed")
session.graph.as_default()
session.close()


# with tf.Session() as session:
#     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     for line in f.readlines()[::-1]:
#         casefile, outcome = line.strip().split()
#         caseid = casefile[:-4]

#         if os.path.isfile('./trainEmbeddings/%s.pickle' % caseid):
#             print(caseid + "already exists, skipped")
#             continue
#         try:
#             outcome = int(outcome)
#             sentences = segmented_sents(caseid)
#             embeddings = session.run(embed(sentences))
#             # embeddings = encoder(sentences, session)
#             with open('./trainEmbeddings/%s.pickle' % caseid, 'wb') as handle:
#                 pickle.dump([embeddings, outcome], handle, protocol=2)
#             print(caseid)
#         except Exception as e:
#             print(caseid + " failed")

