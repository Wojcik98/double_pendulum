docker_double_pendulum_base_image := "double_pendulum_base"
docker_aio_image := "ai_olympics"

[private]
docker-build-double-pendulum-base:
    docker build -t {{docker_double_pendulum_base_image}} \
        {{source_directory()}}/double_pendulum_base
    @echo "Docker image {{docker_double_pendulum_base_image}} built!"

[private]
docker-build-aio: docker-build-double-pendulum-base
    docker build -t {{docker_aio_image}} \
        --build-arg BASE_IMAGE={{docker_double_pendulum_base_image}} \
        {{source_directory()}}/aio
    @echo "Docker image {{docker_aio_image}} built!"
