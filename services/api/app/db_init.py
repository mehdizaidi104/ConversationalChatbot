import json
import os
from . import model  # Use relative import to get the model functions
from pgvector.psycopg2 import register_vector

# The path to the training file *inside the container*
TRAINING_FILE = 'app/trainingDataset.json'

def populate_database(conn):
    """
    Loads intents from JSON, generates embeddings, and inserts
    everything into the PostgreSQL database.
    """
    print(f"Loading training data from: {TRAINING_FILE}")
    
    # 1. Load the JSON file
    try:
        with open(TRAINING_FILE, 'r') as f:
            intents_data = json.load(f)
        
        intents = intents_data.get('intents', [])
        if not intents:
            print("No 'intents' found in the training file.")
            return

    except FileNotFoundError:
        print(f"Error: {TRAINING_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode {TRAINING_FILE}.")
        return

    print(f"Found {len(intents)} intents. Processing...")

    with conn.cursor() as cur:
        # 2. Register the vector type with the connection
        register_vector(cur)

        # 3. Loop through and insert data
        for intent in intents:
            tag = intent.get('tag')
            if not tag:
                continue

            # Insert all responses for this tag
            for response in intent.get('responses', []):
                cur.execute(
                    "INSERT INTO responses (tag, response_text) VALUES (%s, %s)",
                    (tag, response)
                )

            # Insert all patterns and their embeddings for this tag
            for pattern in intent.get('patterns', []):
                # Generate the embedding
                embedding = model.get_embedding(pattern)
                
                cur.execute(
                    "INSERT INTO patterns (tag, pattern_text, embedding) VALUES (%s, %s, %s::vector)",
                    (tag, pattern, embedding)
                )
        
        # 4. Commit all transactions
        conn.commit()
        print("Database population complete.")