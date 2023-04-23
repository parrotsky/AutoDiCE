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

#include "hsplit.h"
#include "cpu.h"

namespace ncnn {

HSplit::HSplit()
{
    one_blob_only = false;
    support_inplace = false;
}

int HSplit::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);          // the split axis. e.g. W(axis=0), H(axis=1), or D(axis=2), C(axis=3).
    starts = pd.get(1, Mat());    // starts = [0,2,3]
    ends = pd.get(2, Mat());   // ends = [23,50] two split points. generate three blobs
    sranks = pd.get(3, Mat());    // sranks = [0,2,3]
    spoints = pd.get(4, Mat());   // spoints = [23,50] two split points. generate three blobs   
    //dpoints = pd.get(3, Mat());   // spoints = [23,50] two split points. generate three blobs
    //hpoints = pd.get(4, Mat());   // spoints = [23,50] two split points. generate three blobs
    //wpoints = pd.get(5, Mat());   // spoints = [23,50] two split points. generate three blobs
    // const int* pa = sranks;
    // fprintf(stderr, "[");
    // for (int i = 0; i < sranks.w; i++)
    // {
    //     fprintf(stderr, " %d", pa[i]);
    // }
    // fprintf(stderr, " ]");
     return 0; 
}
 
template<typename T>
static void copy_cut_image(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src.row<T>(top) + left; // top row, left column.
    T* outptr = dst; //.data;

    for (int y = 0; y < h; y++)
    {
        if (w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w * sizeof(T));
        }
        outptr += w;
        ptr += src.w;
    }
}



int HSplit::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)  const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;  //4u, 16u.    

    fprintf(stderr, "top size %d , get dims %d, w, %d, h %d, c %d\n", top_blobs.size(), dims, w, h, channels);
    int _woffset=0, _hoffset=0, _doffset=0, _coffset=0;
    int _outw=-1, _outh = -1, _outd = -1, _outc;
    // #if NCNN_MPI
    // if(irank == -1){
    //     MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    //     MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    //     fprintf(stderr, "size: %d, rank, %d\n", nrank, irank);      
    // }

    // #endif 
    const int* psranks = sranks;
    const int* pstarts = starts;
    const int* pends = ends;
    const int* pspoints = spoints;
    int group=1;
    //top_blobs.resize(sranks.w);
    fprintf(stderr, "starts %d , %d, \n", pstarts[0], pstarts[1]);
    if (starts.w == sranks.w) group =1;
    if (starts.w == (sranks.w + sranks.w)) group = 2; // grouped convolution.

    if (dims == 1)
    {
        for (int i = 0; i < sranks.w; i++)
        {
            _outw = std::min(pspoints[i], w - pstarts[i]);         
            if (_outw > 0)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, elemsize, opt.blob_allocator);
                if (elemsize == 1)
                    copy_cut_image<signed char>(bottom_blob, top_blob, 0, pstarts[i]);
                if (elemsize == 2)
                    copy_cut_image<unsigned short>(bottom_blob, top_blob, 0, pstarts[i]);
                if (elemsize == 4)
                    copy_cut_image<float>(bottom_blob, top_blob, 0, pstarts[i]);
                //  _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
        return 0;
    }

    if (dims == 2) // C, D, H, W.
    {       
        for (int i = 0; i < sranks.w; i++)
        {
            if (axis == 0) {
                _outh = std::min(pspoints[i], h - _hoffset);
                _woffset = 0; 
                _outw = w;    
            }

            if (axis == 1) {
                _hoffset = 0; 
                _outh = h;
                _outw = std::min(pspoints[i], w - _woffset);;    
            }

            if (_outw > 0 && _outh > 0)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, elemsize, opt.blob_allocator);
                if (elemsize == 1)
                    copy_cut_image<signed char>(bottom_blob, top_blob, _hoffset, _woffset);
                if (elemsize == 2)
                    copy_cut_image<unsigned short>(bottom_blob, top_blob, _hoffset, _woffset);
                if (elemsize == 4)
                    copy_cut_image<float>(bottom_blob, top_blob, _hoffset, _woffset);

                if(axis == 0) _hoffset = std::min(_outh + _hoffset, h);
                if(axis == 1) _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
    }

    if (dims == 3 && group ==1) // C, D, H, W.
    {   
        for (int i = 0; i < sranks.w; i++){
            Mat bottom_blob_sliced = bottom_blob;
            if (axis == 0) { // split c.
                _outc = std::min(pspoints[i], channels  - _coffset);
                _woffset = 0; 
                _outw = w;
                _hoffset = 0; 
                _outh = h; 
                bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);   
            }

            if (axis == 1) {  // split h
                _coffset = 0; 
                _outc = channels;
                _outh = std::min(pends[i],h) - pstarts[i]+1;
                _hoffset = std::min(pstarts[i],h)-1;
                _woffset = 0; 
                _outw = w;                
            }

            if (axis == 2) { // split w.
                _coffset = 0; 
                _outc = channels;
                _hoffset = 0; 
                _outh = h;
                _outw = std::min(pends[i],w) - pstarts[i]+1;
                _woffset = std::min(pstarts[i],w)-1;               
            }

            if (_outw > 0 && _outh > 0 && _outc > 0)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc, elemsize, opt.blob_allocator);
                fprintf(stderr," W %d, H %d, C %d, created.",_outw, _outh, _outc);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 1)
                        copy_cut_image<signed char>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 2)
                        copy_cut_image<unsigned short>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 4)
                        copy_cut_image<float>(m, borderm, _hoffset, _woffset);
                }

                if(axis == 0) _coffset = std::min(_outc + _coffset, channels);
                // if(axis == 1) _hoffset = std::min(_outh + _hoffset, h);
                // if(axis == 2) _woffset = std::min(_outw + _woffset, w);
                }else{
                fprintf(stderr, "Horizontal split value invalid...\n");
                return -100;
                }
            }
    }

    if (dims == 3 && group ==2) // C, D, H, W.
    {       
        channels = channels / 2 ; 
        for (int i = 0; i < sranks.w; i++)
        {
            Mat bottom_blob_sliced = bottom_blob;
            if (axis == 0) { // split c.
                _outc = std::min(pspoints[i], channels - _coffset);
                _woffset = 0; 
                _outw = w;
                _hoffset = 0; 
                _outh = h;    
            }

            if (_outw > 0 && _outh > 0 && _outc > 0)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc * 2,  elemsize, opt.blob_allocator);
                bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);   
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 1)
                        copy_cut_image<signed char>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 2)
                        copy_cut_image<unsigned short>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 4)
                        copy_cut_image<float>(m, borderm, _hoffset, _woffset);
                }

                bottom_blob_sliced = bottom_blob.channel_range(_coffset+channels, _outc);   
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q+_outc);

                    if (elemsize == 1)
                        copy_cut_image<signed char>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 2)
                        copy_cut_image<unsigned short>(m, borderm, _hoffset, _woffset);
                    if (elemsize == 4)
                        copy_cut_image<float>(m, borderm, _hoffset, _woffset);
                }
                if(axis == 0) _coffset = std::min(_outc + _coffset, channels);
            }else{
                return -100;
            }
        }
    }






    if (dims == 4) // C, D, H, W.
    {       
        for (int i = 0; i < sranks.w; i++)
        {
            Mat bottom_blob_sliced = bottom_blob;
            if (axis == 0) { // split c.
                _outc = std::min(pspoints[i], channels - _coffset);
                _doffset = 0; 
                _outd = d; 
                _woffset = 0;
                _outw = w;
                _hoffset = 0; 
                _outh = h; 
                bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);   
            }

            if (axis == 1) {  // split d
                _coffset = 0; 
                _outc = channels;
                _outd = std::min(pspoints[i], d - _doffset);
                _woffset = 0; 
                _outw = w;
                _hoffset = 0; 
                _outh = h;                               
            }

            if (axis == 2) {  // split h
                _doffset = 0; 
                _outd = d; 
                _coffset = 0; 
                _outc = channels;
                _outh = std::min(pspoints[i], h - _hoffset);
                _woffset = 0; 
                _outw = w;                
            }

            if (axis == 3) { // split w.
                _doffset = 0; 
                _outd = d; 
                _coffset = 0; 
                _outc = channels;
                _hoffset = 0; 
                _outh = h;
                _outw = std::min(pspoints[i], w - _woffset);;    
            }        

//       fprintf(stderr, "_outd %d, doffset %d, _outc %d, _coffset  %d, _outh %d, _hoffset:%d, _outw: %d, _woffset: %d elemsize:%d \n", _outd, _doffset,_outc, _coffset, _outh, _hoffset, _outw, _woffset, elemsize);


            if (_outw > 0 && _outh > 0 && _outc > 0 && _outd > 0)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outd, _outc, elemsize, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);

                        if (elemsize == 1)
                            copy_cut_image<signed char>(m, borderm, _hoffset, _woffset);
                        if (elemsize == 2)
                            copy_cut_image<unsigned short>(m, borderm, _hoffset, _woffset);
                        if (elemsize == 4)
                            copy_cut_image<float>(m, borderm, _hoffset, _woffset);
                    }
                }

                if(axis == 0) _coffset = std::min(_outc + _coffset, channels);
                if(axis == 1) _hoffset = std::min(_outd + _doffset, d);
                if(axis == 2) _hoffset = std::min(_outh + _hoffset, h);
                if(axis == 3) _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
    }
    return 0;
}


} // namespace ncnn
