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

#include "receiver.h"

namespace ncnn {

Receiver::Receiver()
{
    #if NCNN_MPI
    nrank = MPI::COMM_WORLD.Get_size();
    irank = MPI::COMM_WORLD.Get_rank();
    #endif
    one_blob_only = true;
}

int Receiver::load_param(const ParamDict& pd)
{
    src_rank = pd.get(0,1);
    size = pd.get(1, 1);  // received data element size: 0;
    node_nums = pd.get(2, 1);

    return 0;
}

int Receiver::load_model(const ModelBin& mb)
{
    node_list = mb.load(node_nums, 1);
    if (node_list.empty())
        return -100;

    for (int i = 0; i < node_nums; i++)
    {
        int _node = static_cast<int>(node_list[i]);
        node_list[i] = _node;
    }

    return 0;
}

/*
int Receiver::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    std::cout << "Receiver run." << std::endl;

    return 0;
}
*/
int Receiver::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_MPI
    // https://www.mpich.org/static/docs/v3.1/www3/MPI_Irecv.html
    MPI_Request requests[node_nums];
    MPI_Status status[node_nums];
    size_t elemsize = bottom_blob.elemsize;
    int actual_size = bottom_blob.total();
    top_blob.create(actual_size, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    float* outptr = top_blob;

    const float* ptr = bottom_blob;
    if (irank==0){
        for (int i = 0; i < node_nums; i++){   
            std::cout<< "sending node: " << node_list[i] << std::endl;
            MPI_Isend(       /* non-blocking send */
                    ptr, actual_size, MPI_FLOAT,       /* triplet of buffer, size, data type */
                    (int)node_list[i],
                    irank,
                    MPI_COMM_WORLD, &requests[i]);       /* send my_int to master */
        }   
   }
 
//    MPI_Wait(&requests[0], &status[0]);

   // for (int i = 0; i < size; i++)
   // {
   //     std::cout << ptr[i]  << " ";
   // }
   // std::cout << std::endl;

   std::vector<float> recv_buff;
   recv_buff.resize(size);
   
   if (irank==1){
    MPI_Irecv(&recv_buff[0], size, MPI_FLOAT, 
            0,      /*  source rank     */ 
            0, 
            MPI_COMM_WORLD, &requests[1]);
    MPI_Wait(&requests[1], &status[1]);

    std::cout<< "recv: " << size << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << recv_buff[i]  << " ";
        outptr[i] =  recv_buff[i];
        
    }
    std::cout << std::endl;

   }

#endif
    return 0;
}

} // namespace ncnn
