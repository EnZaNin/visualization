#!/usr/bin/bash

docker_compose_provider="docker/docker-compose-provider.yaml"

provider_build() {
    docker-compose -f $docker_compose_provider build
}

provider_up() {
    docker-compose -f $docker_compose_provider up -d
}

clear_all() {
  docker stop $(docker ps -a -q)
  docker rm $(docker ps -a -q)
}

main() {
  $1_$2
}

main "$@"