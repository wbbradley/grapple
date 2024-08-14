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
  uuid UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  filename TEXT NOT NULL
);

-- Existence of rows in this relation indicate that this paragraph's triples
-- have been ingested.
-- uuid is based on a SHA256 of the paragraph text.
CREATE TABLE paragraph (
  uuid UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
);

CREATE TABLE triple (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  paragraph_uuid UUID,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  summary TEXT NOT NULL
);

-- CREATE TABLE paragraph_instance (
--   document_uuid UUID PRIMARY KEY,
--   paragraph_uuid UUID PRIMARY KEY,
--   index_span_start BIGINT,
--   index_span_end BIGINT
-- );

COMMIT
