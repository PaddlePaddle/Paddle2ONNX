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

#include "include/deploy/postprocess/ppclas_post_proc.h"

namespace Deploy {

bool PaddleClasPostProc::Run(const std::vector<DataBlob> &outputs,
                            std::vector<PaddleClasResult> *clas_results) {
  clas_results->clear();
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  std::vector<int> output_shape = output_blob.shape;
  int batchsize = output_shape[0];
  int size = 1;
  for (int i = 1; i < output_shape.size(); i++) {
    size *= output_shape[1];
  }
  for (int i = 0; i < batchsize; i++) {
    PaddleClasResult clas_result;
    int fisrt = i * size;
    int last = (i + 1) * size;
    int maxPosition = max_element(
      output_data + fisrt, output_data + last)) - (output_data + fisrt);
    clas_result.class_id = maxPosition;
    clas_result.score = static_cast<double>(output_data[maxPosition]);
  }
}


}  // namespace Deploy
