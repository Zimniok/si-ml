import data_preprocessing
import numpy as np


if __name__ == '__main__':
    x, y = data_preprocessing.get_data(data_preprocessing.GENRES_TO_FILTER)
    print(x)
    print(y)

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    print('\n ANALYZE \n')
    print(data_preprocessing.analyze())

    words_dict = data_preprocessing.make_dict(data_preprocessing.filter_genres(data_preprocessing.read_data(), data_preprocessing.GENRES_TO_FILTER))
    # data_preprocessing.make_encoder(words_dict)

    print('\n MY GENRES \n')

    x2, y2 = data_preprocessing.get_data(data_preprocessing.MY_GENRES)
    print(x2)
    print(y2)

    unique, counts = np.unique(y2, return_counts=True)
    print(dict(zip(unique, counts)))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
