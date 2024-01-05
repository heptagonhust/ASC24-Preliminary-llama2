#!/bin/bash

# This script is used to run the VLLM baseline on the ASC24 cluster over ray.
host=`hostname -s`
if [ "$host" == "$headnode" ]; then
    python llama_offline_inference.py
fi