from sklearn.ensemble import RandomForestClassifier
import keras
import keras.backend as K
import tensorflow as tf
from summarization.type_classifier.vec_features import sent2tensor
from summarization.type_classifier.load import read_files
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


def combine_features_negative_downsample(cases_names, downsample_file):
    x = []
    y = []
    for case_name in cases_names:
        annotated_sentences = read_files([case_name], documents_dir=annotated_dir)

        sentence_tensors = []
        if not os.path.exists(annotated_dir + case_name[:-4] + '.tensor'):
            pure_sentences = [annotated_sentences[i][0] for i in range(len(annotated_sentences))]
            for sent in pure_sentences:
                sentence_tensors.append(sent2tensor(sent))
            # Save sentence_tensors to speed up
            with open(annotated_dir + case_name[:-4] + '.tensor', 'wb') as handle:
                pickle.dump(sentence_tensors, handle, protocol=2)
        else:
            with open(annotated_dir + case_name[:-4] + '.tensor', 'rb') as handle:
                sentence_tensors = pickle.load(handle)

        for i in range(len(annotated_sentences)):
            # Word2Vec feature
            feature1 = sentence_tensors[i]

            # discard if too short
            if feature1 is None:
                continue

            x.append(np.array(feature1))

            sentence, sent_type = annotated_sentences[i]
            if sent_type in finding_types or sent_type in fact_types:  # fact_types
                label = 1
            # elif sent_type in fact_types:
            #     label = 2
            else:
                label = 0
            y.append(label)

    # For downsampled other type sentences
    annotated_sentences = []
    with codecs.open(downsample_dir + downsample_file, 'r', encoding='utf-8') as f:
        for other_sent in f.readlines():
            other_sent = other_sent.strip()
            if len(other_sent) == 0:
                continue
            annotated_sentences.append(other_sent)

    if not os.path.exists(downsample_dir + downsample_file[:-4] + '.tensor'):
        sentence_tensors = []
        for sent in annotated_sentences:
            sentence_tensors.append(sent2tensor(sent))
        # Save embeddings to speed up
        with open(downsample_dir + downsample_file[:-4] + '.tensor', 'wb') as handle:
            pickle.dump(sentence_tensors, handle, protocol=2)
    else:
        with open(downsample_dir + downsample_file[:-4] + '.tensor', 'rb') as handle:
            sentence_tensors = pickle.load(handle)

    for i in range(len(annotated_sentences)):
        # Word2Vec feature
        feature1 = sentence_tensors[i]

        # discard if too short
        if feature1 is None:
            continue

        x.append(np.array(feature1))
        y.append(0)

    print('Number of examples:', len(x))
    return x, y


def list_to_generator(x, y, batch_size=1):
    while True:
        arrX = []
        arrY = []
        i = 1
        for idx in range(len(x)):
            arrX.append(x[idx])
            arrY.append(y[idx])
            if i % batch_size == 0:
                yield (np.expand_dims(np.array(arrX), axis=-1), keras.utils.to_categorical(arrY, num_classes=2))
                # print (np.array(arrX).shape)
                arrX = []
                arrY = []
            i += 1
        if len(arrX) >= 1:
            yield (np.expand_dims(np.array(arrX), axis=-1), keras.utils.to_categorical(arrY, num_classes=2))


class ClassifierCNN():
    def __init__(self):
        # define model
        embedding_dim = 200
        filter_sizes = [2, 3, 4, 5]
        num_filters = 128

        inputs = keras.layers.Input(shape=(None, embedding_dim, 1), dtype='float64')
        conv_0 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[0], embedding_dim),
            padding='valid',
            activation='relu',
            kernel_initializer='normal',
        )(inputs)

        conv_1 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[1], embedding_dim),
            padding='valid',
            activation='relu',
            kernel_initializer='normal',
        )(inputs)
        conv_2 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[2], embedding_dim),
            padding='valid',
            activation='relu',
            kernel_initializer='normal',
        )(inputs)
        conv_3 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[3], embedding_dim),
            padding='valid',
            activation='relu',
            kernel_initializer='normal',
        )(inputs)

        maxpool_0 = keras.layers.GlobalMaxPool2D()(conv_0)
        maxpool_1 = keras.layers.GlobalMaxPool2D()(conv_1)
        maxpool_2 = keras.layers.GlobalMaxPool2D()(conv_2)
        maxpool_3 = keras.layers.GlobalMaxPool2D()(conv_3)

        concat_tensor = keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        # dropout = keras.layers.Dropout(0.2)(concat_tensor)
        # hidden = keras.layers.Dense(units=64, activation='relu')(dropout)
        output = keras.layers.Dense(units=2, activation='softmax')(concat_tensor)

        self._model = keras.models.Model(inputs=inputs, outputs=output)
        self._optimizer = keras.optimizers.Adam(lr=0.001, decay=0.1)
        self._model.compile(loss='categorical_crossentropy', optimizer=self._optimizer, metrics=['accuracy'])
        self._model.summary(line_length=100)
        self._checkpoint = keras.callbacks.ModelCheckpoint(
            filepath='weights.best.hdf5',
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='auto',
            period=1,
        )

    def train(self, train_data, val_data, per_class_metrics, num_epoch):
        self._model.fit_generator(
            generator=train_data,
            steps_per_epoch=945,  # approx. num_examples / batch_size = 3600
            epochs=num_epoch,
            # class_weight={0: 1., 1: 3.},
            use_multiprocessing=True,
            callbacks=[self._checkpoint, per_class_metrics],
            validation_data=val_data,
            validation_steps=338,
        )

    def predict(self, x):
        return self._model.predict(x,  batch_size=1)


class ClassValidation(keras.callbacks.Callback):
    def __init__(self, model, val_x, val_y, target_epoch, total_pred, total_true):
        super(ClassValidation, self).__init__()
        self._model = model
        self._val_x = val_x
        self._val_y = val_y
        self._steps = len(self._val_x)

        self.epoch = 0
        self.target_epoch = target_epoch
        self.total_pred = total_pred
        self.total_true = total_true

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        val_generator = list_to_generator(self._val_x, self._val_y)
        steps = self._steps
        val_trues = []
        val_preds = []

        while steps > 0:
            val_x, val_true = next(val_generator)
            prediction = self.model.predict(val_x)
            val_pred = np.argmax(prediction, axis=-1)
            val_trues.append(np.argmax(val_true, axis=-1).item())
            val_preds.append(val_pred.item())
            steps -= 1
        print("-----------------------------------------------------------")
        print(classification_report(y_true=val_trues, y_pred=val_preds, labels=[0, 1],
                                    target_names=['Others', 'Finding & Fact']))
        print("-----------------------------------------------------------")

        if self.epoch == self.target_epoch:
            self.total_pred += val_preds
            self.total_true += val_trues


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

    model = ClassifierCNN()
    num_epoch = 10
    per_class_validation = ClassValidation(model, x_test, y_test, target_epoch=num_epoch, total_pred=total_prediction,
                                           total_true=total_gold)
    model.train(list_to_generator(x_train, y_train, batch_size=1), list_to_generator(x_test, y_test),
                per_class_validation, num_epoch=num_epoch)

    print("per_class_validation epoch", per_class_validation.epoch)
    print(len(total_prediction))

    print("---------------------ACCUM STATS-------------------------")
    print(classification_report(total_gold, total_prediction, target_names=['Others', 'Finding & Fact']))

"""
num_filters = 64, dropout = 0.2
class_weight={0: 1., 1: 2.}
                precision    recall  f1-score   support

        Others       0.94      0.66      0.77       197
Finding & Fact       0.62      0.92      0.74       119

   avg / total       0.82      0.76      0.76       316


num_filters = 16, dropout = 0.2
class_weight={0: 1., 1: 1.}
                precision    recall  f1-score   support

        Others       0.80      0.80      0.80       197
Finding & Fact       0.67      0.66      0.67       119

   avg / total       0.75      0.75      0.75       316
   

num_filters = 16, dropout = 0.2
class_weight={0: 1., 1: 3.}
                precision    recall  f1-score   support

        Others       0.97      0.61      0.75       197
Finding & Fact       0.60      0.97      0.74       119

   avg / total       0.83      0.74      0.74       316



"""