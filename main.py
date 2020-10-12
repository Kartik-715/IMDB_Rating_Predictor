import pandas as pd
from os import listdir
from os.path import isfile, join

CORPUS_FOLDER = './corpus'
CLEANED_CORPUS_FOLDER = './cleaned_corpus'


def cleanData(data):
    pass


def main():
    FILE_NAMES = [f for f in listdir(CORPUS_FOLDER) if
                  isfile(join(CORPUS_FOLDER, f))]  # Get List of all files in the CORPUS_FOLDER

    for filename in FILE_NAMES:
        imdb_data = pd.read_csv(CORPUS_FOLDER + '/' + filename, sep='\t', header=0)
        print(imdb_data.shape)
        cleanData(imdb_data)
        imdb_data.to_csv(CLEANED_CORPUS_FOLDER + '/' + filename, sep='\t', index=False)


if __name__ == '__main__':
    main()
