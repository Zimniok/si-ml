import data_preprocessing
import numpy as np
import copy as cp
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline


def feature_selection(X, y):
    for mf in [10, 100, 250, 500, 750, None]:
        print('Max features: ' + str(mf))
        pipeline = Pipeline([('vect', CountVectorizer(max_features=mf, stop_words='english', max_df=0.9)), ('nb', MultinomialNB())])
        kfold = KFold(n_splits=10)
        actual_classes, predicted_classes, _ = cross_val_predict(pipeline, kfold, X, y)

        print('Accuracy: %.2f%%' % (accuracy_score(actual_classes, predicted_classes)))
        plot_confusion_matrix(actual_classes, predicted_classes, ["Positive", "Neutral", "Negative"])
        cm = confusion_matrix(actual_classes, predicted_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_preprocessing.GENRES_TO_FILTER)
        disp.plot()
        plt.show()


def cross_val_predict(model, kfold: KFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)

    plt.figure(figsize=(12.8, 6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.show()


if __name__ == '__main__':
    x, y = data_preprocessing.get_data(data_preprocessing.GENRES_TO_FILTER)

    # unique, counts = np.unique(y, return_counts=True)
    # print(dict(zip(unique, counts)))

    # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    X_train, X_validate, X_test = np.split(x, [int(0.6 * len(x)), int(0.8 * len(x))])
    y_train, y_validate, y_test = np.split(y, [int(0.6 * len(x)), int(0.8 * len(x))])

    print(X_test[0])

    feature_selection(x, y)

    # print('Prediction: ' + str(clf.predict(X_test[0])))
    # print('Genre: ' + str(y_test[0]))

    svm = SVC(C=19, kernel='rbf', decision_function_shape='ovo')  # to test C kernel decision_function_shape
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)
    print('Accuracy: ' + str(accuracy_score(y_test, predictions)))
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_preprocessing.GENRES_TO_FILTER)
    disp.plot()
    plt.show()

    test_params = False

    if test_params:
        svc = SVC()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print('Accuracy: ' + str(accuracy_score(y_test, predictions)))

    # print('\n ANALYZE \n')
    # print(data_preprocessing.analyze())
    #
    # # data_preprocessing.make_encoder(words_dict)
    #
    # print('\n MY GENRES \n')
    #
    # x2, y2 = data_preprocessing.get_data(data_preprocessing.MY_GENRES)
    # print(x2)
    # print(y2)
    #
    # unique, counts = np.unique(y2, return_counts=True)
    # print(dict(zip(unique, counts)))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
