#!/bin/bash  
set -x
ar -M < paddle2onnx_bundled-linux-x64.ar.in
ls -lh paddle2onnx-linux-x64/lib/libpaddle2onnx_bundled.a