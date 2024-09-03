#!/bin/bash
# Basic local dev controls for project dbs.
containers=( grapple-graphdb grapple-rdb )
# PG_IMAGE=postgres:16.4
PG_IMAGE='pgvector/pgvector:pg16'

db-init-dbs() {
  if [[ "$1" = '-f' ]]; then
    shift
    docker rm -f "${containers[@]}" 2>/dev/null >&2
    docker wait "${containers[@]}" 2>/dev/null >&2
  fi

  docker run \
    --name grapple-rdb \
    --publish=5432:5432 \
    -e POSTGRES_PASSWORD=postgres \
    -d \
    "$PG_IMAGE"
}

db-rdb-shell() {
  docker run \
    -it \
    --rm \
    --link grapple-rdb:postgres \
    -v "$PWD"/schema.sql:/var/schema.sql \
    -e PGPASSWORD=postgres \
    -e POSTGRES_PASSWORD=irrelevant \
    "$PG_IMAGE" \
    bash
}


db-psql() {
  docker run \
    -it \
    --rm \
    --link grapple-rdb:postgres \
    -e PGPASSWORD=postgres \
    "$PG_IMAGE" \
    psql -h postgres -U postgres "$@"
}

if [[ "$1" = "init" ]]; then
  shift
  db-init-dbs "$@"
elif [[ "$1" = "shell" ]]; then
  shift
  db-rdb-shell
elif [[ "$1" = "psql" ]]; then
  shift
  db-psql "$@"
fi
