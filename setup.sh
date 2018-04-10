#!/usr/bin/env bash

# install dependencies
apt-get install -y protobuf-compiler libprotoc-dev

pip install -r requirements.txt

if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi
