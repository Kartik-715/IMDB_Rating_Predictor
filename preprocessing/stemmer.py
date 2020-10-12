import pandas as pd
import nltk

def porter_stemmer(data):
    ps = nltk.porter.PorterStemmer()
    final_data = pd.DataFrame(columns=['Text', 'Rating'])
    for index, row in data.iterrows():
        text = row[0]
        final_text = ' '.join([ps.stem(word) for word in text.split()])
        final_data = final_data.append({'Text': final_text, 'Rating': row[1]}, ignore_index = True)
    return final_data
