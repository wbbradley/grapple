#!/bin/bash
# Basic local dev controls for project dbs.
containers=( grapple-graphdb grapple-rdb )

db-init-dbs() {
  if [[ "$1" = '-f' ]]; then
    shift
    docker rm -f "${containers[@]}"
    docker wait "${containers[@]}"
  fi

  docker run \
    --name grapple-graphdb \
    --restart always \
    --publish=7474:7474 \
    --publish=7687:7687 \
    --env NEO4J_AUTH=none \
    -d \
    neo4j:5.22.0

  docker run \
    --name grapple-rdb \
    --publish=5432:5432 \
    -e POSTGRES_PASSWORD=postgres \
    -d \
    postgres:16.4
}

db-rdb-shell() {
  docker run \
    -it \
    --rm \
    --link grapple-rdb:postgres \
    -v "$PWD"/schema.sql:/var/schema.sql \
    -e PGPASSWORD=postgres \
    -e POSTGRES_PASSWORD=irrelevant \
    postgres:16.4 \
    bash
}

db-psql() {
  docker run \
    -it \
    --rm \
    --link grapple-rdb:postgres \
    -e PGPASSWORD=postgres \
    postgres:16.4 \
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
