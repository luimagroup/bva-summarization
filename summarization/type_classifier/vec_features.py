from gensim.models.word2vec import Word2Vec
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

#
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
input_sentences = tf.placeholder(tf.string, shape=[None])
embedding = embed(input_sentences)

# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../law2vec-seg/law2vecv4')
# nltk_stopwords = set(stopwords.words('english') + list(string.punctuation))
# law2vec = Word2Vec.load(model_path)


# Tokenize a sentence to a list of words and remove stopwords
def preprocess(sentence):
    words = []
    content = nltk.word_tokenize(sentence.lower())
    for j in range(len(content)):
        word = content[j]
        # if word in nltk_stopwords:
        #     continue
        # if len(word) == 1:
        #     continue
        words.append(word)

    return words


# def word2vec(sentence):
#     words = preprocess(sentence)
#     sent_vec = None
#     count = 0
#
#     for word in words:
#         try:
#             word_vector = law2vec[word]
#             if not sent_vec:
#                 sent_vec = np.zeros(word_vector.shape[0])
#             sent_vec += word_vector
#             count += 1
#         except:
#             continue
#
#     sent_vec /= count
#     return sent_vec.tolist()


# def sent2tensor(sentence):
#     words = preprocess(sentence)
#     sent_tensor = []
#     count = 0
#
#     for word in words:
#         try:
#             word_vector = law2vec[word]
#             sent_tensor.append(word_vector)
#             count += 1
#         except:
#             continue
#
#     if len(sent_tensor) < 5:
#         return None
#
#     return np.vstack(sent_tensor)


def word2vec(sentences):
    session = tf.InteractiveSession()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = embedding.eval(feed_dict={input_sentences: sentences})
    session.graph.as_default()
    session.close()
    return embeddings


if __name__ == '__main__':
    pass
    # print(sent2tensor("This is a PTSD case.").shape)
    # print(sent2tensor("This is a PTSD case."))
    # print(word2vec("hello cool."))

