#!/bin/bash

# Define your volume, container name, and image
VOLUME_NAME="elasticsearch-data"
CONTAINER_NAME="es-rag01"
IMAGE_NAME="dataanalyse-docker-local.artifactory.ccta.dk/mlemba/elasticsearch:8.15.3"

# Check if the container exists
if podman ps -a --filter name="${CONTAINER_NAME}" | grep -q "${CONTAINER_NAME}"; then
    echo "Container '${CONTAINER_NAME}' already exists. Pruning..."
    podman container rm -f ${CONTAINER_NAME}
else
    echo "Container '${CONTAINER_NAME}' does not exist."
fi

# Start the container with the provided command
echo "Starting container '${CONTAINER_NAME}'..."
podman run --name "${CONTAINER_NAME}" -p 9200:9200 -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -d "${IMAGE_NAME}"