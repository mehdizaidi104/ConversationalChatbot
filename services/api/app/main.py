import os
import random
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from pgvector.psycopg2 import register_vector
import psycopg2
from . import model, schemas, db_init # Import other modules
from psycopg2 import pool

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global db_pool
    DATABASE_URL = os.environ['DATABASE_URL']
    
    print("Connecting to database...")
    db_pool = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
    
    print("Loading embedding model...")
    model.load_model() # Loads the Sentence-Transformer model into memory
    
    # Check if DB is populated, if not, run the init script
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM patterns LIMIT 1;")
            if cur.fetchone() is None:
                print("Database is empty. Populating with training data...")
                db_init.populate_database(conn)
                print("Database populated successfully.")
        db_pool.putconn(conn)
    except Exception as e:
        print(f"Error checking/populating database: {e}")
        # This will fail if tables don't exist, so init.sql must run first
        
    print("API startup complete.")
    yield
    # --- Shutdown ---
    if db_pool:
        db_pool.closeall()
    print("API shutdown complete.")

# Helper function for dependency injection
def get_db_conn():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# --- FastAPI App ---
app = FastAPI(title="Chatbot API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Chatbot API is running"}

@app.post("/predict", response_model=schemas.QueryResponse)
def predict(query: schemas.QueryInput, conn=Depends(get_db_conn)):
    try:
        # 1. Embed the user's query
        query_embedding = model.get_embedding(query.text)
        
        # 2. Find the closest matching pattern in the DB
        with conn.cursor() as cur:
            register_vector(cur) # Register pgvector type
            # The '<=>' operator calculates cosine distance
            cur.execute(
                "SELECT tag FROM patterns ORDER BY embedding <=> %s::vector LIMIT 1",
                (query_embedding,)
            )
            result = cur.fetchone()
            
            if not result:
                return schemas.QueryResponse(response_text="Sorry, I don't understand that.")

            best_tag = result[0]
            
            # 3. Get a random response for that tag
            cur.execute(
                "SELECT response_text FROM responses WHERE tag = %s ORDER BY RANDOM() LIMIT 1",
                (best_tag,)
            )
            response = cur.fetchone()
            
            if not response:
                return schemas.QueryResponse(response_text="I know about that topic, but I'm speechless.")

            return schemas.QueryResponse(response_text=response[0])

    except Exception as e:
        print(f"Prediction Error: {e}")
        return schemas.QueryResponse(response_text="Sorry, something went wrong on my end.")