import streamlit as st
import torch
import random
import json
import pickle
from spacy_utils import sentence_embedding, tokenize_and_normalize
from model import NeuralNet

# Cache the resources
@st.cache_resource
def load_model_and_embeddings():
    """
    Loads the model and pre-saved GloVe embeddings from disk.
    """
    # Load the saved model and metadata
    FILE = "model_and_metadata.pth"
    saved_data = torch.load(FILE, map_location=torch.device("cpu"))

    # Extract metadata
    emb_size = saved_data["emb_size"]
    hidden_size = saved_data["hidden_size"]
    num_classes = saved_data["num_classes"]
    tags = saved_data["tags"]
    model_state = saved_data["model_state"]

    # Recreate the model
    model = NeuralNet(emb_size, hidden_size, num_classes)
    model.load_state_dict(model_state)
    model.eval()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load pre-saved GloVe embeddings
    with open("glove_embeddings.pkl", "rb") as f:
        word2vec = pickle.load(f)

    return model, device, word2vec, tags, emb_size

@st.cache_data
def load_intents():
    """
    Loads intents from the training dataset JSON file.
    """
    with open("trainingDataset.json", "r") as f:
        return json.load(f)["intents"]

# Load resources
model, device, word2vec, tags, emb_size = load_model_and_embeddings()
intents = load_intents()

# Define chatbot response function
def get_response(sentence):
    """
    Generates a chatbot response based on the input sentence.
    """
    tokens = tokenize_and_normalize(sentence, ["?", ",", ".", "!"])
    emb_vec = sentence_embedding(tokens, word2vec, emb_size)
    X = torch.from_numpy(emb_vec).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X)
        _, predicted_idx = torch.max(output, dim=1)
        predicted_tag = tags[predicted_idx.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted_idx.item()]

    if prob.item() > 0.75:
        for intent in intents:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])
    return "Sorry, I don't understand your query."

# Streamlit UI
st.title("Chatbot with GloVe Embeddings")
user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    response = get_response(user_input)
    st.write(f"Chatbot: {response}")
