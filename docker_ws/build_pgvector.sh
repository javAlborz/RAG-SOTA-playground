#!/bin/bash
# This script is used to install the required packages for the project
# It is used by the CI/CD pipeline to install the required packages for the project
podman build -t postgres_rag:v0.1 .