import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.layers import MaxPooling2D
from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import cross_val_score, KFold
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


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

predictions = model.predict(X_test)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test.values

labels = ['2', '3', '5']
cm = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)  # shape=(12, 12)

print(cm)

plot_confusion_matrix(cm, labels, title='CNN Confusion Matrix', cmap='magma', normalize=False)


