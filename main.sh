#!/usr/bin/bash

docker_compose_provider="docker-compose-dev.yml"

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

provider_stop() {
  docker-compose -f $docker_compose_provider down
}

provider_restart() {
  clear_all
  provider_build
  provider_up
}

main() {
  $1_$2
}

main "$@"