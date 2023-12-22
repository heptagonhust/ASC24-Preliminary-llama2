#!/bin/bash

host=`hostname -s`
if [ "$host" == "$headnode" ]; then
    ray start --head --num-gpus 2 --node-ip-address $headaddr --port=$port
else
    sleep 5
    ray start --num-gpus 2 --address="$headaddr:$port"
fi
