#!/bin/bash

# install libpng needed by opencv
if [ -x "$(command -v  ldconfig -p | grep png16 )" ];then
    apt-get install libpng16-16
fi

# install libjasper1 needed by opencv
if [ -x "$(command -v  ldconfig -p | grep libjasper )" ];then
    if [ ! -x "$(command -v  add-apt-repository )"]; then
        apt-get install software-properties-common
    fi
    add-apt-repository “deb http://security.ubuntu.com/ubuntu xenial-security main”
    apt update
    apt install libjasper1 libjasper-dev
fi

