CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS target_entities (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    primary_contact JSONB,
    all_contacts JSONB,
    raw_content TEXT,
    embedding VECTOR(3072),
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Backfill columns on databases created before contact persistence was added
ALTER TABLE target_entities ADD COLUMN IF NOT EXISTS primary_contact JSONB;
ALTER TABLE target_entities ADD COLUMN IF NOT EXISTS all_contacts JSONB;

CREATE INDEX ON target_entities USING hnsw (embedding vector_cosine_ops);