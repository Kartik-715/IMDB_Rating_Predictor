from preprocessing.denoise import denoise_text
from preprocessing.stemmer import porter_stemmer
from preprocessing.stop import toktok_nltk_stop

def rename_columns(data):
    data = data.fillna(' ')
    data['Text'] = data['Comment Head'].str.cat(data['Comment Body'], sep=" ")
    data.rename(columns={'Comment Rating': 'Rating'}, inplace=True)
    del data['Comment Head']
    del data['Comment Body']
    cols = data.columns.tolist()
    cols = cols[::-1]
    data = data[cols]
    return data

def preprocess_data(data):
    data = rename_columns(data)
    data.dropna(inplace=True)
    data = denoise_text(data)
    data = porter_stemmer(data)
    data = toktok_nltk_stop(data)
    return data
