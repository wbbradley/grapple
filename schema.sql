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

COMMIT;

        WITH nearest_embeddings AS (
          SELECT uuid
          FROM embedding
          ORDER BY vector <-> "hey"
          LIMIT 29
        ), enriched_triples AS (
          SELECT
            t.id,
            t.created_at,
            p.text AS paragraph_text,
            es.text AS subject_text,
            ep.text AS predicate_text,
            eo.text AS object_text,
            esu.text AS summary_text
          FROM triple t
          LEFT JOIN paragraph p ON t.paragraph_uuid = p.uuid
          LEFT JOIN embedding es ON t.subject_uuid = es.uuid
          LEFT JOIN embedding ep ON t.predicate_uuid = ep.uuid
          LEFT JOIN embedding eo ON t.object_uuid = eo.uuid
          LEFT JOIN embedding esu ON t.summary_uuid = esu.uuid
        )
        SELECT *
        FROM enriched_triples et
        JOIN nearest_embeddings ne ON et.summary_uuid = ne.uuid
        ORDER BY et.created_at
