#!/bin/bash

#set -ex 
source "$(dirname "$0")/device_ip.sh"

scp -P 8022 $1 u0_a241@${device_ip}:/data/data/com.termux/files/home/workspace/test/mnn/cnn_test/$2
