import csv
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt


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
    date_string = 3
    genre = 4
    lyrics = 5

    all_lyrics = []
    all_dates = []

    fifties = 0
    sixties = 0
    seventies = 0
    eighties = 0
    nineties = 0
    two_thousands = 0
    twenty_tens = 0

    for row in readCSV:
        # to adjust how much data is being used, change this number
        # there are approximately 28000 lines, so using 30 gives us ~ 28000 / 30 = 900 songs
        if i % 30 == 0:
            date = int(row[date_string])
            if date > 1949 and date < 1960:
                fifties += 1
            elif date > 1959 and date < 1970:
                sixties += 1
            elif date > 1969 and date < 1980:
                seventies += 1
            elif date > 1979 and date < 1990:
                eighties += 1
            elif date > 1989 and date < 2000:
                nineties += 1
            elif date > 1999 and date < 2010:
                two_thousands += 1
            elif date > 2009 and date < 2020:
                twenty_tens += 1
            else:
                print(date)

        i += 1

    plt.style.use('seaborn')
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    years = ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s']
    songs = [fifties, sixties, seventies, eighties,
             nineties, two_thousands, twenty_tens]
    ax.bar(years, songs)
    plt.show()
    plt.savefig('dates2.png')
    """
    # creating the dataset
    data = {'1950s': fifties, '1960s': sixties, '1970s': seventies, '1980s': eighties,
            '1990s': nineties, '2000s': two_thousands, '2010s': twenty_tens}
    decades = list(data.keys())
    songs = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(decades, songs, color='maroon',
            width=0.4)

    plt.xlabel("Decade")
    plt.ylabel("Number of Songs")
    plt.title("Distribution of Songs by Decade")
    plt.savefig('dates.png')


read_csv()
