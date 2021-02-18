// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include <map>

namespace Deploy {

class DataBlob{
  public:
    // data
    std::vector<char> data;
    
    // data name
    std::string name;

    // data shape
    std::vector<int> shape;

    /* 
    data dtype
    0: FLOAT32
    1: INT64
    2: INT32
    3: UINT8
    */
    int dtype;

    //LoD信息
    std::vector<std::vector<size_t>> lod;

};

class ShapeInfo{
  public:
    
    // shape trace
    std::vector<std::vector<int> > shape;
    
    // transform order
    std::vector<std::string> transform_order;
   
};

}//namespace
