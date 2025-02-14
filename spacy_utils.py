import spacy
import numpy as np
import pickle

nlp = spacy.load("en_core_web_sm")

def load_glove_embeddings(emb_path, emb_dim=300):
    """
    Loads GloVe vectors from a .txt file and returns a dictionary.
    If embeddings are already saved, loads them from the pickle file.
    """
    try:
        with open("glove_embeddings.pkl", "rb") as f:
            print("Loading pre-saved GloVe embeddings...")
            return pickle.load(f)
    except FileNotFoundError:
        print("Building GloVe embeddings...")
        word2vec = {}
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                vals = line.strip().split()
                word = vals[0]
                vec = np.array(vals[1:], dtype=np.float32)
                if len(vec) == emb_dim:
                    word2vec[word] = vec
        with open("glove_embeddings.pkl", "wb") as f:
            pickle.dump(word2vec, f)
        return word2vec

def sentence_embedding(tokens, word2vec, emb_dim=300):
    """
    Converts a list of tokens into an average embedding using GloVe-like dictionary.
    """
    vectors = [word2vec[token] for token in tokens if token in word2vec]
    if len(vectors) == 0:
        return np.zeros(emb_dim, dtype=np.float32)
    return np.mean(vectors, axis=0)

def tokenize_and_normalize(text, ignore_chars=['?', '.', ',', '!']):
    """
    Tokenize a string using spaCy and remove specified characters.
    """
    doc = nlp(text.lower())
    tokens = [t.text for t in doc if t.text not in ignore_chars]
    return tokens
