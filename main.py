import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


def show_image(vec):
    # send train_data.iloc[0, :].values
    img = vec.reshape(16, 16)
    plt.imshow(img, cmap='plasma')
    plt.show()


def load_data():
    train = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.train.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)
    test = pd.read_csv("/Users/shany.shmuely/Documents/Rscripts/zip.test.gz", compression='gzip', header=None, sep=' ', quotechar='"', error_bad_lines=False)
    train = train.loc[train.iloc[:, 0].isin([2, 3, 5])]
    test = test.loc[test.iloc[:, 0].isin([2, 3, 5])]
    return train, test


def print_covariance_diff_between_classes(train):
    train_only_two = train.loc[train.iloc[:, 0] == 2].iloc[:, 1:-1]
    train_only_three = train.loc[train.iloc[:, 0] == 3].iloc[:, 1:-1]
    train_only_five = train.loc[train.iloc[:, 0] == 5].iloc[:, 1:-1]

    # each class's samples does not have the same covariance, therefore the LDA assumptions do not hold and
    # covariance that is used is sum_k prior_k * C_k
    print('Diff (norm) between classes covariance')
    print(LA.norm(train_only_two.cov() - train_only_three.cov()))
    print(LA.norm(train_only_two.cov() - train_only_five.cov()))
    print(LA.norm(train_only_three.cov() - train_only_five.cov()))


def separate_data_labels(train, test):
    train_digits = train.iloc[:, 0]
    train_data = train.iloc[:, 1:-1]
    test_digits = test.iloc[:, 0]
    test_data = test.iloc[:, 1:]
    return train_data, train_digits, test_data, test_digits


def run_KNN(train_data, train_digits, test_data, test_digits):
    k_values = [1, 5, 10, 20]
    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k, p=2)
        neigh.fit(train_data, train_digits)

        # train
        plot_confusion_matrix(neigh, train_data, train_digits)
        plt.title(f"Train score: {round(neigh.score(train_data, train_digits), 2)}")
        plt.savefig(f'KNN_results/cm_train_{k}-NN.png')

        # test
        plot_confusion_matrix(neigh, test_data, test_digits)
        plt.title(f"Test score: {round(neigh.score(test_data, test_digits), 2)}")
        plt.savefig(f'KNN_results/cm_test_{k}-NN.png')


def run_LDA(train_data, train_digits, test_data, test_digits):
    lda = LDA(store_covariance=True)
    lda.fit(train_data, train_digits)
    # train
    plot_confusion_matrix(lda, train_data, train_digits)
    plt.title(f"Train score: {round(lda.score(train_data, train_digits), 2)}")
    plt.savefig(f'LDA_results/cm_train.png')

    # test
    plot_confusion_matrix(lda, test_data, test_digits)
    plt.title(f"Test score: {round(lda.score(test_data, test_digits), 2)}")
    plt.savefig(f'LDA_results/cm_test.png')


def run_LDA_dim_reduction(train_data, train_digits, test_data, test_digits):
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

    lda = LDA(store_covariance=True)
    lda.fit(pc_train, train_digits)

    plot_confusion_matrix(lda, pc_train, train_digits)
    plt.title(f"Train score: {round(lda.score(pc_train, train_digits), 2)}")
    plt.savefig(f'LDA_results/cm_dim_reduction_train.png')

    plot_confusion_matrix(lda, pc_test, test_digits)
    plt.title(f"Test score: {round(lda.score(pc_test, test_digits), 2)}")
    plt.savefig(f'LDA_results/cm_dim_reduction_test.png')


def show_misclassified_images(test_digits, predicted_test, test_data):
    # predicted 3 actual 5
    misclassifications_indices = np.where((test_digits - predicted_test) == 2)
    for i in misclassifications_indices[0]:
        show_image(test_data.iloc[i, :].values)


def run_logistic_regression(train_data, train_digits, test_data, test_digits):
    lr = LogisticRegression(fit_intercept=True, class_weight='balanced', max_iter=1000,
                            solver='lbfgs', multi_class='multinomial', n_jobs=1)

    lr.fit(train_data, train_digits)
    predicted_test = lr.predict(test_data)
    print(classification_report(test_digits, predicted_test))

    show_misclassified_images(test_digits, predicted_test, test_data)

    # train
    plot_confusion_matrix(lr, train_data, train_digits, cmap='plasma')
    plt.title(f"Train score: {round(lr.score(train_data, train_digits), 2)}")
    plt.savefig(f'LogitR_results/cm_train.png')

    # test
    plot_confusion_matrix(lr, test_data, test_digits, cmap='plasma')
    plt.title(f"Test score: {round(lr.score(test_data, test_digits), 2)}")
    plt.savefig(f'LogitR_results/cm_test.png')


def run_decision_tree(train_data, train_digits, test_data, test_digits):
    classifier = DecisionTreeClassifier(criterion="gini", max_features="sqrt", class_weight="balanced")
    # classifier = DecisionTreeClassifier(criterion="entropy", max_features="sqrt", class_weight="balanced")
    classifier.fit(train_data, train_digits)
    print(classification_report(test_digits, classifier.predict(test_data)))

    # train
    plot_confusion_matrix(classifier, train_data, train_digits, cmap='plasma')
    plt.title(f"Train score: {round(classifier.score(train_data, train_digits), 2)}")
    plt.savefig(f'DT_results/cm_train.png')

    # test
    plot_confusion_matrix(classifier, test_data, test_digits, cmap='plasma')
    plt.title(f"Test score: {round(classifier.score(test_data, test_digits), 2)}")
    plt.savefig(f'DT_results/cm_test.png')


def main():
    train, test = load_data()
    train_data, train_digits, test_data, test_digits = separate_data_labels(train, test)

    # run_KNN(train_data, train_digits, test_data, test_digits)
    print_covariance_diff_between_classes(train)
    # run_LDA(train_data, train_digits, test_data, test_digits)
    # run_LDA_dim_reduction(train_data, train_digits, test_data, test_digits)
    # run_logistic_regression(train_data, train_digits, test_data, test_digits)
    # run_decision_tree(train_data, train_digits, test_data, test_digits)

    # tree
    # svm

    print('DONE')


if __name__ == "__main__":
    main()