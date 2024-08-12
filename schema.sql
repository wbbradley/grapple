BEGIN;
CREATE TABLE embedding (
    id SERIAL PRIMARY KEY,
    sentence_text TEXT NOT NULL,
    model TEXT NOT NULL,
    vector JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX ix_embedding ON embedding (sentence_text, model);

CREATE TABLE provenance (
    id SERIAL PRIMARY KEY,
    embedding_sentence_text TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    document_name TEXT NOT NULL,
    sentence_position INT NOT NULL,
    FOREIGN KEY (embedding_sentence_text, embedding_model) REFERENCES embedding(sentence_text, model)
);
COMMIT
