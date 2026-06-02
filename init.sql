CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS target_entities (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    raw_content TEXT,
    
    embedding VECTOR(3072),
    
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON target_entities USING hnsw (embedding vector_cosine_ops);