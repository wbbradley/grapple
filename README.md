# grapple

## Setup

```bash
docker run \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=none \
    neo4j:5.22.0
```

Access via http://localhost:7474/browser/.

NB: auth is disabled in dev.

More docs at https://neo4j.com/docs/operations-manual/current/docker/introduction/.
