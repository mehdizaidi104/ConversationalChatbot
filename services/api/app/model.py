from sentence_transformers import SentenceTransformer

# We'll use this specific model. It matches the 384 dimensions in your init.sql
MODEL_NAME = 'all-MiniLM-L6-v2'
model = None

def load_model():
    """
    Loads the Sentence-Transformer model into memory.
    This is called once on API startup in main.py.
    """
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

def get_embedding(text: str):
    """
    Generates an embedding for a given text string.
    """
    if model is None:
        raise Exception("Model is not loaded. Call load_model() first.")
    
    # .encode() is the method to create the vector embedding
    return model.encode(text)