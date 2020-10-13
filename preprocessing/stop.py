import nltk
import pandas as pd
import numpy as np

def toktok_nltk_stop(data, is_lower_case=False):
    tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
    nltk.download('stopwords')
    stopword_list = nltk.corpus.stopwords.words('english')
    final_data = pd.DataFrame(columns=['Text', 'Rating'])
    for index, row in data.iterrows():
        text = row[0]
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token == 'not' or token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() == 'not' or token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        final_data = final_data.append({'Text': filtered_text, 'Rating': row[1]}, ignore_index = True)
    return final_data
