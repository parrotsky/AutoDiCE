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

#include "mylayer.h"

namespace ncnn {

MyLayer::MyLayer()
{
    #if NCNN_MPI
    nrank = MPI::COMM_WORLD.Get_size();
    irank = MPI::COMM_WORLD.Get_rank();
    #endif
    one_blob_only = true;
    support_inplace = true;
}

int MyLayer::load_param(const ParamDict& pd)
{

    return 0;
}

int MyLayer::load_model(const ModelBin& mb)
{

    return 0;
}

int MyLayer::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    std::cout << "MyLayer run." << std::endl;

    return 0;
}

int MyLayer::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    //int a = get_world_id();
#if NCNN_MPI
    int number;
    if (irank == 0) {
        number = 145;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (irank == 1) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n",
                number);
    }

    printf( "mylayer is not implement.irank: %d out from %d processors \n", irank, nrank);    
#endif
    return 0;
}

} // namespace ncnn
