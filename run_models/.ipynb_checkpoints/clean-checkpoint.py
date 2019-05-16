import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


def clean_salary(i):
    i = i.replace('k','').replace('M','').replace('$','').replace('CA','').replace('£','').replace('–','–').split('–')
    return (i)

def avg(i):
    if type(i) != float:
        try:
            average = (float(clean_salary(i)[0]) + float(clean_salary(i)[1])) / 2
        except:
            average = float(clean_salary(i)[0])
        return average *1000
    else:
        return str(np.nan)
    
def token_stop_lemm(corpus,stop_words):
    strin = ''
    tokens = word_tokenize(corpus)
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    lemmatized = [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]
    for word in lemmatized:
        strin+= ' '+word
    return strin
