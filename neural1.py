import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
np.random.seed(1671)

NB_EPOCH = 30
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = RMSprop()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3


(x_train, y_train), (x_test, y_test) = mnist.load_data()
RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# M_HIDDEN скрытых слоев
# 10 выходов
# на последнем этапе softmax
model = Sequential()

model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER, metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score: ", score[0])
print('Test accuracy: ', score[1])

# # построить график изменения верности
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('accuracy_MPS_RMS.png')
# plt.clf()
# # plt.show()
#
# # построить график изменения потери
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('loss_MPS_RMS.png')
# # plt.show()

# Записываем модель в файл
model_json = model.to_json()
json_file = open("mnist_model_MPS_DP.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("model_MPS_DP.h5")