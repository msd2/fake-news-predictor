import re
from functools import reduce


def deEmojify(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)




def remove_hyperlinks(text):
    text = re.sub(r"http\S+", "", text)
    return text

def remove_punctuation(text):
    text = re.sub(r'[)(|!%?@#*,/:;…-]', ' ', text)
    text = re.sub(r'&amp', 'and', text)
    return text

def fix_apostrophes(text):
    text = re.sub(r'’', "'", text)
    return text

def remove_whitespace(text):
    text = re.sub(r'[\t\n\r]', ' ', text)
    return text

def normalize_whitespace(text):
    text = re.sub(r' +', ' ', text)
    return text

def strip_space(text):
    return text.strip()

def end_with_fullstop(text):
#     print(text)
    if len(text)>0 and text[-1]=='.':
        return text
    else:
        return text + '.'

def string_process(text):
    func_list = [
        deEmojify,
        remove_hyperlinks,
        remove_punctuation,
        fix_apostrophes,
        remove_whitespace,
        normalize_whitespace,
        strip_space,
        end_with_fullstop
    ]
    text = reduce(lambda x, func: func(x), func_list, text)
    return text.lower()