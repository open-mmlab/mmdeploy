#!/bin/bash
docker_image=$1
codebase_list=($2)

for codebase in ${codebase_list[@]}
do
    echo $codebase 
done