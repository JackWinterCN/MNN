#!/bin/bash

#set -ex 
source "$(dirname "$0")/device_ip.sh"
ssh u0_a241@${device_ip} -p 8022
