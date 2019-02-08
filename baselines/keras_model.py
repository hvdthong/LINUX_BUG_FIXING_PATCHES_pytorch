from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import os
from keras.models import Sequential


def lstm_cnn_all_data(x_train, y_train, FLAGS):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    model = Sequential()
    model.add(Embedding(FLAGS.vocab_msg, FLAGS.msg_length))
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
    model.add(LSTM(FLAGS.hidden_units))
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

    filepath = outputFolder + "/lstm_cnn" + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                                 save_best_only=False, save_weights_only=True, \
                                 mode='auto', period=1)
    callbacks_list = [checkpoint]

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs, callbacks=callbacks_list)
    return model
