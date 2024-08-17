CREATE EXTENSION IF NOT EXISTS vector;

BEGIN;

CREATE TABLE embedding (
  uuid UUID PRIMARY KEY,
  text TEXT NOT NULL,
  model TEXT NOT NULL,
  vector VECTOR(3072) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX ix_embedding ON embedding (text, model);

CREATE TABLE tag (
  id BIGSERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE embedding_tag (
  embedding_uuid UUID NOT NULL,
  tag_id BIGINT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE UNIQUE INDEX ix_embedding_tag ON embedding_tag (embedding_uuid, tag_id);

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
  text TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE triple (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  paragraph_uuid UUID,
  subject_uuid UUID NOT NULL,
  predicate_uuid UUID NOT NULL,
  object_uuid UUID NOT NULL,
  summary_uuid UUID NOT NULL
);

-- CREATE TABLE paragraph_instance (
--   document_uuid UUID PRIMARY KEY,
--   paragraph_uuid UUID PRIMARY KEY,
--   index_span_start BIGINT,
--   index_span_end BIGINT
-- );

COMMIT
