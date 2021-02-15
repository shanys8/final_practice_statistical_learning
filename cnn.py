from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import pandas as pd
from keras.layers import MaxPooling2D
from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import cross_val_score, KFold
from keras.models import load_model


def convert_y_class(y):
    y.loc[(y == 2)] = 0
    y.loc[(y == 3)] = 1
    y.loc[(y == 5)] = 2
    return y


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(16, 16, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

    # output layer
    model.add(Dense(3, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    current_acc = 0
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        if acc > current_acc:
            model.save('final_model')
        current_acc = acc
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# Load data
train = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.train.gz", compression='gzip', header=None, sep=' ',
                    quotechar='"', error_bad_lines=False)
test = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.test.gz", compression='gzip', header=None, sep=' ',
                   quotechar='"', error_bad_lines=False)
train = train.loc[train.iloc[:, 0].isin([2, 3, 5])]
test = test.loc[test.iloc[:, 0].isin([2, 3, 5])]
y_train = train.iloc[:, 0]
trainX = train.iloc[:, 1:-1]
y_test = test.iloc[:, 0]
testX = test.iloc[:, 1:]
X_train = trainX.values.reshape((trainX.shape[0], 16, 16, 1))
X_test = testX.values.reshape((testX.shape[0], 16, 16, 1))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 3
y_train = convert_y_class(y_train)
y_test = convert_y_class(y_test)

Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# Evaluate model
scores, histories = evaluate_model(X_train, Y_train)
# learning curves
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)

model = load_model('final_model')
# evaluate model on test dataset
_, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %.3f' % (acc * 100.0))

