-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for patterns and their embeddings
CREATE TABLE patterns (
    id SERIAL PRIMARY KEY,
    tag VARCHAR(255) NOT NULL,
    pattern_text TEXT NOT NULL,
    embedding vector(384) -- 384 is the dimension of the 'all-MiniLM-L6-v2' model
);

-- Create a table for responses
CREATE TABLE responses (
    id SERIAL PRIMARY KEY,
    tag VARCHAR(255) NOT NULL,
    response_text TEXT NOT NULL
);