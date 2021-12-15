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
        # if i > 16000 and i < 17000:
        if i % 30 == 0:
            all_lyrics.append(row[lyrics])
            all_dates.append(int(row[date]))
            # print(row[date])

        # print(row[lyrics])
        i += 1
        # if i > 50:
        #   break

    # convert to numpy arrays
    x_lyrics = np.array(all_lyrics)
    y_dates = np.array(all_dates)
    return (x_lyrics, y_dates)


# print(x_lyrics)
read_csv()
