import data_reader
import numpy as np


if __name__ == '__main__':
    x, y = data_reader.get_data(data_reader.GENRES_TO_FILTER)
    print(x)
    print(y)

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    print('\n ANALYZE \n')
    print(data_reader.analyze())

    data_reader.make_dict(data_reader.filter_genres(data_reader.read_data(), data_reader.GENRES_TO_FILTER))

    # print('\n MY GENRES \n')
    #
    # x2, y2 = data_reader.get_data(data_reader.MY_GENRES)
    # print(x2)
    # print(y2)
    #
    # unique, counts = np.unique(y2, return_counts=True)
    # print(dict(zip(unique, counts)))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
