// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

#define USE_P2O_MAPPER(op_name__, class_name__)    \
  extern op_name##Generator* op_name##inst;        \
  int P2O_MAPPER_REGISTER_FAKE(op_name__) UNUSED = \
  op_name##inst->Touch##op_name##class_name(); 

#define P2O_MAPPER_REGISTER_FAKE(op_name__) op_name__##__registry__