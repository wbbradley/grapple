# grapple

An exploration of RAG.

## Setup

```bash
# Set up the local environment.
$ ./setup-env -f
# Initialize the related Docker containers.
$ ./dbs init -f
# Run the DDL to create the local cache.
$ grapple migrate
# Ingest something:
$ grapple ingest some-large-text-file.txt
# Query:
$ grapple query
```
