import 'docker/docker.just'

alias db := docker-build
alias xs := xhost-stuff
alias da := docker-attach

default:
    @just --list

docker-start:
    #!/bin/bash
    ./.devcontainer/xhost_stuff.sh
    docker compose --file docker/compose.yaml up -d aio

docker-stop:
    docker compose --file docker/compose.yaml down

[doc("Should be executed from host, not inside docker.")]
docker-attach:
    @docker exec -it aio_dev bash -l

[doc("Should be executed from host, not inside docker.")]
docker-build: docker-build-aio
    @echo "All docker images built!"

[doc("Should be executed from host, not inside docker.")]
xhost-stuff:
    @sudo xhost +local:docker
    @sudo xhost +
