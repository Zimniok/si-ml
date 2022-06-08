import csv
import json
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


GENRES_TO_FILTER = ['Science Fiction', 'Fantasy', 'Mystery', 'Historical novel']
MY_GENRES = ['Fantasy', 'Mystery', 'Historical novel', 'Horror']

CHARS_TO_REMOVE = '!?.,:;)('


def get_data(genres_to_filter, number_of_features, max_df):
    data = read_data()
    data = filter_genres(data, genres_to_filter)
    # output = filter_stopwords(output)
    # print(output)

    data = np.array(data)
    data = np.transpose(data)
    vectorizer = CountVectorizer(max_features=number_of_features, stop_words='english', max_df=max_df)
    vectorized = vectorizer.fit_transform(data[1])
    return vectorized, np.array(data[0]), vectorizer


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
            if genre in row[0]:
                output_genre.append(genre)
        if len(output_genre) == 1:
            output.append([GENRES_TO_FILTER.index(output_genre[0]), row[1]])
    return output


def filter_stopwords(data):
    output = []
    for book in data:
        stop_words = set(stopwords.words('english'))
        word_tokens = (w for w in re.split(r"\W", book[1]) if w)
        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        output.append((book[0], filtered_sentence))
    return output


def to_x_y(data):
    x = []
    y = []
    for row in data:
        x.append(row[1])
        print(row[0])
        y.append(GENRES_TO_FILTER.index(row[0]))
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



# def make_encoder(words_dict):
#     enc = OneHotEncoder(categories=words)
#     print(enc.categories_)
