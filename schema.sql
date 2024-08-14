BEGIN;
CREATE TABLE embedding (
  id SERIAL PRIMARY KEY,
  sentence_text TEXT NOT NULL,
  model TEXT NOT NULL,
  vector JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX ix_embedding ON embedding (sentence_text, model);

CREATE TABLE document (
  id SERIAL PRIMARY KEY,
  content_uuid UUID PRIMARY KEY,
  filename TEXT NOT NULL
);

-- Existence of rows in this relation indicate that this paragraph's triples
-- have been ingested.
-- uuid is based on a SHA256 of the paragraph text.
CREATE TABLE paragraph (
  uuid UUID PRIMARY KEY
);

CREATE TABLE triple (
  uuid TEXT PRIMARY KEY,
  paragraph_uuid UUID PRIMARY KEY,
  subject TEXT,
  predicate TEXT,
  object TEXT,
  summary TEXT
);

CREATE TABLE paragraph_instance (
  document_id UUID PRIMARY KEY,
  paragraph_uuid UUID PRIMARY KEY,
  index_span_start BIGINT,
  index_span_end BIGINT
);

COMMIT
