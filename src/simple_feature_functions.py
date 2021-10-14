import re
import spacy
nlp = spacy.load('en_core_web_sm')

def punct_count(text):
    _, count = re.subn(r'\W', '', text)
    return count

def token_count(text):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct]
    return len(tokens)

def text_length(text):
    return len(text)

def average_token_length(text):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct]
    sum_ = 0
    counter = 0
    for token in tokens:
        sum_+=len(token)
        counter+=1
    return sum_/counter


def apply_text_functions(df, col):
    funcs = [
        punct_count,
        token_count,
        text_length
#         average_token_length
    ]
    for func in funcs:
        df[col+'_'+func.__name__] = df[col].apply(func)
    return df