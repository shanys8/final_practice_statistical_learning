import numpy as np
import pandas as pd;  pd.set_option('display.max_columns', None, 'display.max_rows', 2000)
from datetime import datetime, date
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
from sklearn.metrics import plot_confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

def load_data():
    train = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.train.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)
    test = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.test.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)

    train = train.loc[train.iloc[:, 0].isin([2, 3, 5])]
    test = test.loc[test.iloc[:, 0].isin([2, 3, 5])]

    train_digits = train.iloc[:, 0]
    train_data = train.iloc[:, 1:-1]

    test_digits = test.iloc[:, 0]
    test_data = test.iloc[:, 1:]

    return train_data, train_digits, test_data, test_digits


def main():
    train_data, train_digits, test_data, test_digits = load_data()

    # KNN
    k_values = [1, 5, 10, 20]
    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k, p=2)
        neigh.fit(train_data, train_digits)
        plot_confusion_matrix(neigh, test_data, test_digits)
        plt.title(f"Train score: {round(neigh.score(train_data, train_digits), 2)}\nTest score: {round(neigh.score(test_data, test_digits), 2)}")
        plt.savefig(f'knn_results/cm_{k}-nn.png')




    # show image
    # img = train_data.iloc[0, :].values.reshape(16, 16)
    # plt.imshow(img)
    # plt.show()



    print('DONE')


if __name__ == "__main__":
    main()