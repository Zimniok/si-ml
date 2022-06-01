import csv
import json
import numpy as np

GENRES_TO_FILTER = ['Science Fiction', 'Fantasy', 'Mystery', 'Historical novel']
MY_GENRES = ['Fantasy', 'Mystery', 'Historical novel', 'Horror']

CHARS_TO_REMOVE = '!?.,:;)('


def get_data(genres_to_filter):
    data = read_data()
    output = filter_genres(data, genres_to_filter)
    # print(output)
    return to_x_y(output)


def read_data():
    output = []
    with open('./booksummaries/booksummaries.txt', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            genres = []
            if len(row[5]) != 0:
                genres = json.loads(row[5])
                genres = list(genres.values())
            output.append([genres, row[6]])
    return output


def filter_genres(data, genres):
    output = []
    for row in data:
        output_genre = []
        for genre in genres:
            if row[0].__contains__(genre):
                output_genre.append(genre)
        if len(output_genre) == 1:
            output.append([''.join(output_genre), row[1]])
    return output


def to_x_y(data):
    x = []
    y = []
    for row in data:
        # x.append(row[1])
        y.append(row[0])
    return np.array(x), np.array(y)


def analyze():
    data = read_data()
    output = {}
    for row in data:
        for genre in row[0]:
            if genre in output:
                output[genre] = output[genre] + 1
            else:
                output[genre] = 1
    return dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
    # Fantasy Mystery Historical novel Horror


def make_dict(filtered_data):
    counts = {}  # word: (genres, books count)
    for row in filtered_data:
        dsc = row[1]
        for char in CHARS_TO_REMOVE:
            dsc = dsc.replace(char, '')
        words = dsc.split(' ')
        for word in words:
            if word == '':
                continue
            word = word.lower()
            if word in counts:
                if row[0] not in counts[word][0]:
                    counts[word][0].append(row[0])
                counts[word] = counts[word][0], counts[word][1] + 1
            else:
                counts[word] = [row[0]], 1

    counts = {k: v for k, v in counts.items() if len(v[0]) < 4}

    print(dict(sorted(counts.items(), key=lambda item: item[1][1])))
