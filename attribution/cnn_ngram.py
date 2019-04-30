import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf
import os
import pickle
from sklearn.metrics import classification_report


def generate_arrays_from_folder(path, batch_size):
    while True:
        arrX = []
        arrY = []
        i = 1
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'rb') as f:
                x, y = pickle.load(f)
                arrX.append(x)
                arrY.append(y + 1)
            if i % batch_size == 0:
                yield (np.expand_dims(np.array(arrX), axis=-1), keras.utils.to_categorical(arrY, num_classes=3))
                # print (np.array(arrX).shape)
                arrX = []
                arrY = []
            i += 1
        if len(arrX) >= 1:
            yield (np.expand_dims(np.array(arrX), axis=-1), keras.utils.to_categorical(arrY, num_classes=3))


class ClassValidation(keras.callbacks.Callback):
    def __init__(self, model, val_data_path, steps):
        super(ClassValidation, self).__init__()
        self.model = model
        self.val_data_path = val_data_path
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        val_generator = generate_arrays_from_folder(self.val_data_path, batch_size=1)
        steps = self.steps
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
        print(classification_report(y_true=val_trues, y_pred=val_preds, labels=[0, 1, 2],
                                    target_names=['denied', 'remanded', 'granted']))
        print("-----------------------------------------------------------")


def train():
    gpu_num = 1
    batch_size = 1
    data_path = "../sentence_encoder/trainEmbeddings"
    dir_name = data_path[data_path.rfind('/') + 1:]
    tmp_dir = 'tmp' + dir_name[0].upper() + dir_name[1:]
    if os.path.exists(tmp_dir):
        data_path = tmp_dir
    val_data_path = "../sentence_encoder/valEmbeddings"
    dir_name = val_data_path[val_data_path.rfind('/') + 1:]
    tmp_dir = 'tmp' + dir_name[0].upper() + dir_name[1:]
    if os.path.exists(tmp_dir):
        val_data_path = tmp_dir
    test_data_path = "../sentence_encoder/testEmbeddings"

    # define model
    embedding_dim = 512
    filter_sizes = [2, 3, 4]
    num_filters = 256

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

    maxpool_0 = keras.layers.GlobalMaxPool2D()(conv_0)
    maxpool_1 = keras.layers.GlobalMaxPool2D()(conv_1)
    maxpool_2 = keras.layers.GlobalMaxPool2D()(conv_2)

    concat_tensor = keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    hidden = keras.layers.Dense(units=64, activation='relu')(concat_tensor)
    output = keras.layers.Dense(units=3, activation='softmax')(hidden)

    model = keras.models.Model(inputs=inputs, outputs=output)
    if gpu_num > 1:
        model = keras.utils.multi_gpu_model(model, gpus=gpu_num)

    model.summary(line_length=100)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='weights.best.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='auto',
        period=1,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=3,
        verbose=0,
        mode='auto',
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    per_class_validation = ClassValidation(model, val_data_path, steps=956)
    model.fit_generator(
        generator=generate_arrays_from_folder(data_path, batch_size),
        steps_per_epoch=14428,
        epochs=20,
        # class_weight={0: 1., 1: 20., 2: 2.},
        use_multiprocessing=True,
        callbacks=[checkpoint, early_stopping, per_class_validation],
        validation_data=generate_arrays_from_folder(val_data_path, batch_size),
        validation_steps=956,
    )


if __name__ == '__main__':
    train()
