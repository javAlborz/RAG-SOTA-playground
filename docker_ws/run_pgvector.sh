#!/bin/bash

# create a Docker volume for PostgreSQL data
# check if pgdata volume exists and create if it does not
podman volume ls | grep pgdata
if [ $? -ne 0 ]; then
    podman volume create pgdata
fi

# check if rag-postgres container exists and prune if it does
podman ps -a | grep rag-postgres
if [ $? -eq 0 ]; then
    podman container rm -f rag-postgres
fi

# start the PostgreSQL container with the Docker volume
podman run \
  --name rag-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v pgdata:/var/lib/postgresql/data \
  --net=host \
  -d localhost/postgres_rag:v0.1