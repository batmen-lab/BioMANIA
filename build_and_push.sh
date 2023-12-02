#!/bin/bash

# set version
version="v1.1.7"
# set name for repository
repository="chatbotuibiomania/biomania-together"

# iterloop all the folders under dir
for dir in ./docker_utils/*/; do
    # get folder name as LIB
    lib=$(basename "$dir")

    # build
    echo "Building Docker image for $lib"
    sudo docker build --build-arg LIB="$lib" -t "$repository:$version-$lib-cuda12.1-ubuntu22.04" -f Dockerfile ./

    # push
    echo "Pushing Docker image for $lib"
    sudo docker push "$repository:$version-$lib-cuda12.1-ubuntu22.04"
done

