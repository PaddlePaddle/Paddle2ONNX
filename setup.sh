#!/usr/bin/env bash

# install python dependencies
if [ -f "requirements.txt" ]; then
    apt-get install -y protobuf-compiler libprotoc-dev
    pip install -r requirements.txt
fi
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi
