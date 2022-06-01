import csv
import json
import numpy as np

GENRES_TO_FILTER = ['Satire', 'Roman', 'Science Fiction']


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
    output = filter_genres(output, GENRES_TO_FILTER)
    return to_x_y(output)


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
        x.append(row[1])
        y.append(row[0])
    return np.array(x), np.array(y)
