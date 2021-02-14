import numpy as np
import pandas as pd;  pd.set_option('display.max_columns', None, 'display.max_rows', 2000)
from datetime import datetime, date
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def show_image(vec):
    # send train_data.iloc[0, :].values
    img = vec.reshape(16, 16)
    plt.imshow(img)
    plt.show()


def load_data():
    train = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.train.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)
    test = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.test.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)

    train = train.loc[train.iloc[:, 0].isin([2, 3, 5])]
    test = test.loc[test.iloc[:, 0].isin([2, 3, 5])]

    return train, test


def main():
    train, test = load_data()

    train_digits = train.iloc[:, 0]
    train_data = train.iloc[:, 1:-1]
    test_digits = test.iloc[:, 0]
    test_data = test.iloc[:, 1:]

    train_only_two = train.loc[train.iloc[:, 0] == 2].iloc[:, 1:-1]
    train_only_three = train.loc[train.iloc[:, 0] == 3].iloc[:, 1:-1]
    train_only_five = train.loc[train.iloc[:, 0] == 5].iloc[:, 1:-1]

    # each class's samples does not have the same covariance, therefore the LDA assumptions do not hold and
    # covariance that is used is sum_k prior_k * C_k
    print('Diff between classes covariance')
    print(LA.norm(train_only_two.cov() - train_only_three.cov()))
    print(LA.norm(train_only_two.cov() - train_only_five.cov()))
    print(LA.norm(train_only_three.cov() - train_only_five.cov()))


    # KNN
    # k_values = [1, 5, 10, 20]
    # for k in k_values:
    #     neigh = KNeighborsClassifier(n_neighbors=k, p=2)
    #     neigh.fit(train_data, train_digits)
    #
    #     # train
    #     plot_confusion_matrix(neigh, train_data, train_digits)
    #     plt.title(f"Train score: {round(neigh.score(train_data, train_digits), 2)}")
    #     plt.savefig(f'KNN_results/cm_train_{k}-NN.png')
    #
    #     # test
    #     plot_confusion_matrix(neigh, test_data, test_digits)
    #     plt.title(f"Test score: {round(neigh.score(test_data, test_digits), 2)}")
    #     plt.savefig(f'KNN_results/cm_test_{k}-NN.png')


    # # LDA
    # lda = LDA(store_covariance=True)
    # lda.fit(train_data, train_digits)
    # # train
    # plot_confusion_matrix(lda, train_data, train_digits)
    # plt.title(f"Train score: {round(lda.score(train_data, train_digits), 2)}")
    # plt.savefig(f'LDA_results/cm_train.png')
    #
    # # test
    # plot_confusion_matrix(lda, test_data, test_digits)
    # plt.title(f"Test score: {round(lda.score(test_data, test_digits), 2)}")
    # plt.savefig(f'LDA_results/cm_test.png')

    # LDA dimensionality reduction

    # feature scaling
    sc = StandardScaler()
    train_data_scaled = sc.fit_transform(train_data)
    test_data_scaled = sc.transform(test_data)

    # select the 20% pc that are the most significant
    u, s, vh = LA.svd(train_data_scaled, full_matrices=True)
    pc_num = len(s[s > np.percentile(s, 80)])
    pca = PCA(n_components=pc_num)
    pc_train = pca.fit_transform(train_data_scaled)
    pc_test = pca.fit_transform(test_data_scaled)

    lda_dim_reduction = LDA(store_covariance=True)
    lda_dim_reduction.fit(pc_train, train_digits)

    plot_confusion_matrix(lda_dim_reduction, pc_train, train_digits)
    plt.title(f"Train score: {round(lda_dim_reduction.score(pc_train, train_digits), 2)}")
    plt.savefig(f'LDA_results/cm_dim_reduction_train.png')

    plot_confusion_matrix(lda_dim_reduction, pc_test, test_digits)
    plt.title(f"Test score: {round(lda_dim_reduction.score(pc_test, test_digits), 2)}")
    plt.savefig(f'LDA_results/cm_dim_reduction_test.png')

    print('DONE')


if __name__ == "__main__":
    main()