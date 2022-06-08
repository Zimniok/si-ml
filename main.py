import data_preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


if __name__ == '__main__':
    x, y, vectorizer = data_preprocessing.get_data(data_preprocessing.GENRES_TO_FILTER, 500, 0.9)
    print(x)
    print(vectorizer.get_feature_names_out())
    print(y)

    # unique, counts = np.unique(y, return_counts=True)
    # print(dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    clf = MultinomialNB()

    clf.fit(X_train, y_train)

    print('Prediction: ' + str(clf.predict(X_test[0])))
    print('Genre: ' + str(y_test[0]))

    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_preprocessing.GENRES_TO_FILTER)
    disp.plot()
    plt.show()

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
