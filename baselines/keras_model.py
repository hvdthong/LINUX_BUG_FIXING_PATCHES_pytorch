from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Sequential


def lstm_cnn_model(dictionary_size, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(dictionary_size, FLAGS.embedding_dim))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    # -------------------------------------------
    # model.add(LSTM(lstm_output_size))
    # model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(LSTM(FLAGS.hidden_units, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(FLAGS.hidden_units, activation="relu"))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(Dense(1, activation='sigmoid'))
    return model


def lstm_cnn_all_data(x_train, y_train, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(FLAGS.vocab_msg, FLAGS.embedding_dim))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    # -------------------------------------------
    # model.add(LSTM(lstm_output_size))
    # model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(LSTM(FLAGS.hidden_units, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(FLAGS.hidden_units, activation="relu"))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    outputFolder = './keras_model'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    filepath = outputFolder + "/lstm_cnn_all" + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                                 save_best_only=False, save_weights_only=True, \
                                 mode='auto', period=1)
    callbacks_list = [checkpoint]

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs, callbacks=callbacks_list)
    return model


def lstm_cnn_split_data(x_train, y_train, x_test, y_test, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(FLAGS.vocab_msg, FLAGS.embedding_dim))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    # -------------------------------------------
    # model.add(LSTM(lstm_output_size))
    # model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(LSTM(FLAGS.hidden_units, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(FLAGS.hidden_units, activation="relu"))
    model.add(Dropout(FLAGS.dropout_keep_prob))
    # -------------------------------------------
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    outputFolder = './keras_model'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    filepath = outputFolder + "/lstm_cnn_split" + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                                 save_best_only=False, save_weights_only=True, \
                                 mode='auto', period=1)
    callbacks_list = [checkpoint]

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test), callbacks=callbacks_list)
    return model
