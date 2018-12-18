# coding:utf-8
from __future__ import print_function
import os
import os.path
import json
from sklearn.metrics import classification_report
import sys
import time
import numpy as np
from keras.utils import np_utils
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D, LSTM, GRU
from keras.models import Model
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import data_preprocessing as dp

from keras import optimizers, backend
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.preprocessing import text, sequence
from keras.models import load_model

class TextClassifier(object):
    def __init__(self, conf_path, ispredict):
        try:
            param = json.load(open(conf_path))
        except Exception as e:
            print("%s, parameter load ERROR!"%(conf_path))
            print(e)
            sys.exit(0)
        self.clean_dataset_path = param["clean_dataset_path"]
        self.input_dataset_path = param["input_dataset_path"]
        self.words_path = param["words_path"]
        self.weights_path = param["weights_path"]
        self.model_path = param["model_path"]
        self.category2id_path = param["category2id_path"]
        self.maxlen = param["maxlen"]
        self.batch_size = param["batch_size"]
        self.nb_epoch = param["nb_epoch"]
        self.embedding_length = param["embedding_length"]
        self.gpuid = param["gpuid"]
        self.max_features = 0
        self.split_rate = param["split_rate"]
        self.nb_classes = param["nb_classes"]
        self.word2id = {}
        self.category2id = {}
        self.id2category = {}
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        if ispredict == 1:
            self.model = self.load_model()

    def preprocess(self):
        print("preprocess data...")
        df_dataset, self.nb_classes = dp.load_data_from_disk(self.input_dataset_path, self.clean_dataset_path)
        print("generate and save words...")
        self.word2id = dp.generate_words(df_dataset)
        dp.save_words(self.word2id, self.words_path)
        self.max_features = len(self.word2id)
        print("generate and save category2id...")
        self.category2id = dp.generate_category2id(df_dataset)
        dp.save_category2id(self.category2id, self.category2id_path)
        x_train, y_train, x_test, y_test, self.maxlen = dp.split_train_test(df_dataset,self.word2id,self.split_rate,self.category2id)
        # padding
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        y_train = np_utils.to_categorical(np.array(y_train), num_classes=self.nb_classes)
        y_test = np_utils.to_categorical(np.array(y_test), num_classes=self.nb_classes)
        print("max_features: %d" % self.max_features)
        print("max len: %d" % self.maxlen)
        print("x_train:", x_train.shape)
        print("x_test:", x_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)
        return x_train, y_train, x_test, y_test


    # CNN Model
    def create_model(self):
        max_features = self.max_features + 1  # input dims
        inputs = Input(shape=(self.maxlen,))
        embed = Embedding(max_features, self.embedding_length)(inputs)
        conv_3 = Conv1D(filters=256, kernel_size=3, padding="valid", activation="relu", strides=1)(embed)
        conv_4 = Conv1D(filters=256, kernel_size=4, padding="valid", activation="relu", strides=1)(embed)
        conv_5 = Conv1D(filters=256, kernel_size=5, padding="valid", activation="relu", strides=1)(embed)
        pool_3 = GlobalMaxPooling1D()(conv_3)
        pool_4 = GlobalMaxPooling1D()(conv_4)
        pool_5 = GlobalMaxPooling1D()(conv_5)
        cat = Concatenate()([pool_3, pool_4, pool_5])
        output = Dropout(0.25)(cat)
        dense1 = Dense(256, activation='relu')(output)
        bn = BatchNormalization()(dense1)
        dense = Dense(self.nb_classes, activation='softmax')(bn)
        model = Model(inputs=inputs, outputs=dense)
        return model

    def evaluation(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        row_max = [np.where(x==np.max(x))[0][0] for x in y_pred]
        y_pred = np.zeros(shape=(len(row_max), self.nb_classes))
        for row, col in enumerate(row_max):
            y_pred[row, col] = 1.0
        target_names = ['class ' + str(i) for i in range(0, self.nb_classes)]
        y_test = [np.where(x==np.max(x))[0][0] for x in y_test]
        y_pred = [np.where(x==np.max(x))[0][0] for x in y_pred]
        cm = confusion_matrix(y_test, y_pred)
        print('confusion matrix: \n')
        print(cm)
        print("classification report:")
        print(classification_report(y_test, y_pred, target_names=target_names))


    def train(self):
        """ load dataset and train model """
        x_train, y_train, x_test, y_test = self.preprocess()
        """ GPU parameter """
        with tf.device('/gpu:' + str(self.gpuid)):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
            tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                             log_device_placement=True,
                                             gpu_options=gpu_options))
            model = self.create_model()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            print("model summary")
            model.summary()
            print("checkpoint_dir: %s" % self.model_path+'/'+'checkpoint.h5')
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                         ModelCheckpoint(self.model_path+'/'+'checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
                         ]
            print("training started...")
            tic = time.process_time()
            model.fit(x_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.nb_epoch,
                        validation_data=(x_test, y_test),
                        shuffle=1,
                        callbacks= callbacks)
            toc = time.process_time()
            print("training ended...")
            print("Total Computation time: " + str((toc - tic) / 60) + " mins ")
            model.save(self.weights_path)
            backend.set_learning_phase(0)
            sess = backend.get_session()
            ts = time.time()
            builder = tf.saved_model.builder.SavedModelBuilder(self.model_path+'/'+str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))))
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
            builder.save()
            self.evaluation(model, x_test, y_test)
            print("Completed!")

    backend.clear_session()
    tf.reset_default_graph()

