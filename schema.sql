CREATE EXTENSION IF NOT EXISTS vector;

-- NB: all UUIDs are content-addressable keys based on a SHA256 of the related
-- text AND/OR a superkey of related fields. See str_to_uuid, etc..

BEGIN;

CREATE TABLE document (
  uuid UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  filename TEXT NOT NULL
);

-- Existence of rows in this relation indicate that this paragraph's triples
-- have been ingested.
CREATE TABLE paragraph (
  uuid UUID PRIMARY KEY, -- hash(document_uuid, span_index..)
  text TEXT NOT NULL,
  document_uuid UUID NOT NULL,
  span_index_start BIGINT NOT NULL,
  span_index_lim BIGINT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE UNIQUE INDEX ix_paragraph ON paragraph (document_uuid, span_index_start, span_index_lim);
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


CREATE TABLE triple (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  paragraph_uuid UUID,
  subject_uuid UUID NOT NULL,
  predicate_uuid UUID NOT NULL,
  object_uuid UUID NOT NULL,
  summary_uuid UUID NOT NULL
);

COMMIT;
