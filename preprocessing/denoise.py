from bs4 import BeautifulSoup
import pandas as pd
import re

def strip_html(data):
    final_data = pd.DataFrame(columns=['Text', 'Rating'])
    for index, row in data.iterrows():
        text = row[0]
        soup = BeautifulSoup(text, "html.parser")
        final_text = soup.get_text()
        final_data = final_data.append({'Text': final_text, 'Rating': row[1]}, ignore_index = True)
    return final_data

def remove_between_square_brackets(data):
    final_data = pd.DataFrame(columns=['Text', 'Rating'])
    for index, row in data.iterrows():
        text = row[0]
        final_text = re.sub('\[[^]]*\]', '', text)
        final_data = final_data.append({'Text': final_text, 'Rating': row[1]}, ignore_index = True)
    return final_data

def remove_special_characters(data, remove_digits=True):
    final_data = pd.DataFrame(columns=['Text', 'Rating'])
    pattern=r'[^a-zA-z0-9\s]'
    for index, row in data.iterrows():
        text = row[0]
        final_text = re.sub(pattern,'',text)
        final_data = final_data.append({'Text': final_text, 'Rating': row[1]}, ignore_index = True)
    return final_data

def denoise_text(data):
    data = strip_html(data)
    data = remove_between_square_brackets(data)
    data = remove_special_characters(data)
    return data
