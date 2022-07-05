import re

def preprocess_text(text, stemmer):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([stemmer.stem(w) for w in text.split()])
    return text