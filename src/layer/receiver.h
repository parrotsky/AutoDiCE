// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_RECEIVER_H
#define LAYER_RECEIVER_H

#include "layer.h"
#include <mpi.h>

namespace ncnn {

class Receiver : public Layer
{
public:
    Receiver();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

//    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_top_blob, Mat& top_blob, const Option& opt) const;

public:
//    // param
    int src_rank;
    int size;      // communication data size
    int node_nums;
    Mat node_list; // communication nodes number list

//    // model
//    Mat scale_data;
//    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_SCALE_H
