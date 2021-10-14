import spacy

nlp = spacy.load('en_core_web_sm')

# def tokenize_string(text):
#     all_stopwords = nlp.Defaults.stop_words
#     tokens = nlp(text)
#     tokens = [token for token in tokens if not token.is_punct]
#     # tokens = [token for token in tokens if token.text not in ["'s",'s']]
#     tokens = [token.lemma_ for token in tokens]
#     tokens = [token for token in tokens if token not in all_stopwords]
#     return tokens


def tokenize_string(text):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_stop]
    tokens = [token for token in tokens if token.pos_ not in ['PUNCT','SYM','NUM','PART','SPACE']]
    tokens = [token for token in tokens if token.text not in [
        "n't","'h",'m','wh','%','rt',"'s","'ve","'ll",'’re',
        "'m",'&',"'ve","'re",'’ve','’ll','’s','’m','n’t','s.','c.','f.','m.'
    ]]
    tokens = [token.lemma_ for token in tokens]
    return tokens