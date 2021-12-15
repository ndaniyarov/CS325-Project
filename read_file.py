import csv
import os.path
import sys
import numpy as np


def read_csv():
    # open and read file
    if os.path.exists('music-data.csv'):
        csvfile = open('music-data.csv')
        readCSV = csv.reader(csvfile, delimiter=',')
    else:
        print("File not found.")
        sys.exit()

    # skip first line
    next(readCSV)

    i = 0

    name = 2
    date = 3
    genre = 4
    lyrics = 5

    all_lyrics = []
    all_dates = []

    for row in readCSV:
        # to adjust how much data is being used, change this number
        # there are approximately 28000 lines, so using 30 gives us ~ 28000 / 30 = 900 songs
        if i % 30 == 0:
            all_lyrics.append(row[lyrics])
            all_dates.append(int(row[date]))

        i += 1

    # convert to numpy arrays
    x_lyrics = np.array(all_lyrics)
    y_dates = np.array(all_dates)
    return (x_lyrics, y_dates)
