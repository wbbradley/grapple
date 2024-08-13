BEGIN;
CREATE TABLE embedding (
    id SERIAL PRIMARY KEY,
    sentence_text TEXT NOT NULL,
    model TEXT NOT NULL,
    vector JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX ix_embedding ON embedding (sentence_text, model);

COMMIT
