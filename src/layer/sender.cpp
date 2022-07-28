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

#include "sender.h"

namespace ncnn {

Sender::Sender()
{
    #if NCNN_MPI

    MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

//    nrank = MPI::COMM_WORLD.Get_size();
//    irank = MPI::COMM_WORLD.Get_rank();
    #endif
    one_blob_only = true;
    support_inplace = true;
}

int Sender::load_param(const ParamDict& pd)
{
    // dest_rank = pd.get(0, 0);
    size = pd.get(0, 1);
    node_nums = pd.get(1, 1);
    return 0;
}

int Sender::load_model(const ModelBin& mb)
{
    node_list = mb.load(node_nums, 1);
    if (node_list.empty())
        return -100;
    // static cast float to int32.
    for (int i = 0; i < node_nums; i++)
    {
        int _node = static_cast<int>(node_list[i]);
        node_list[i] = _node;
    }

    return 0;
}


/*

int Sender::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    std::cout << "Sender run." << std::endl;

    return 0;
}
*/
int Sender::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    //    int w = bottom_top_blob.;
    //    int h = bottom_top_blob.h;
    //    int channels = bottom_top_blob.c;
    //    int size = w * h; // 每一个通道上有size个单元
#if NCNN_MPI
    MPI_Request requests[node_nums];
    int total_size = bottom_top_blob.total(); // 输入张量的单元总数
    int send_size = (total_size < size)? total_size : size;

    float* ptr = bottom_top_blob;
    for (int i = 0; i < node_nums; i++){   
        std::cout<< "sending node: " << node_list[i] << std::endl;
        MPI_Isend(       /* non-blocking send */
                ptr, send_size, MPI_FLOAT,       /* triplet of buffer, size, data type */
                (int)node_list[i],
                irank,
                MPI_COMM_WORLD, &requests[i]);       /* send my_int to master */
    } 

#endif
    return 0;
}

} // namespace ncnn
