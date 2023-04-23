// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "hsplit_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

HSplit_arm::HSplit_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __ARM_NEON
static void crop_pack8_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const float* ptr = src.row(top) + left * 8;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            vst1q_f32(outptr, _p0);
            vst1q_f32(outptr + 4, _p1);
            ptr += 8;
            outptr += 8;
        }

        ptr += (left + right) * 8;
    }
}

static void crop_pack8_bf16_fp16s_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const unsigned short* ptr = src.row<unsigned short>(top) + left * 8;
    unsigned short* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            vst1q_u16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }

        ptr += (left + right) * 8;
    }
}

static void crop_pack4_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const float* ptr = src.row(top) + left * 4;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}

static void crop_pack4_bf16_fp16s_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const unsigned short* ptr = src.row<unsigned short>(top) + left * 4;
    unsigned short* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            uint16x4_t _p = vld1_u16(ptr);
            vst1_u16(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}
#endif // __ARM_NEON
 
int HSplit_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int ref_elempack = reference_blob.elempack;

    Mat& top_blob = top_blobs[0];
    int _woffset=0, _hoffset=0, _doffset=0, _coffset=0;
    int _outw=-1, _outh = -1, _outd = -1, _outc;
    const int* psranks = sranks;
    const int* pstarts = starts;
    const int* pends = ends;
    const int* pspoints = spoints;
    int group=1;
    //top_blobs.resize(sranks.w);
    fprintf(stderr, "starts %d , %d, \n", pstarts[0], pstarts[1]);
    if (starts.w == sranks.w) group =1;
    if (starts.w == (sranks.w + sranks.w)) group = 2; // grouped convolution.



#if __ARM_NEON
    if (elempack == 8)
    {
        if (dims == 1)
        {

            for (int i = 0; i < sranks.w; i++){
                _outw = std::min(pspoints[i], w - pstarts[i]); 

                int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
                size_t out_elemsize = elemsize / elempack * out_elempack;
            
                if (_outw > 0 && _woffset % 8 == 0 && out_elempack == 8)
                {
                    ncnn::Mat& top_blob = top_blobs[i];
                    top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                    if (elemsize == 16u)
                        crop_pack8_bf16_fp16s_neon(bottom_blob, top_blob, 0, pstarts[i] / elempack);
                    else
                        crop_pack8_neon(bottom_blob, top_blob, 0, pstarts[i] / elempack);
                    //  _woffset = std::min(_outw + _woffset, w);
                }else{
                    return -100;
                }
            }
                return 0;
        }

        if (dims == 2)
        {
            // pack h dimension.

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

            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _hoffset % 8 == 0 && out_elempack == 8)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elemsize == 16u)
                    crop_pack8_bf16_fp16s_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack8_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                if(axis == 0) _hoffset = std::min(_outh + _hoffset, h);
                if(axis == 1) _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
        return 0;
        
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
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
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

            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _outc > 0 && _coffset % 8 == 0 && out_elempack == 8)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                // fprintf(stderr," W %d, H %d, C %d, created.",_outw, _outh, _outc);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
                    if (elemsize == 16u)
                        crop_pack8_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack8_neon(m, borderm, _hoffset, _woffset);
                }

                if(axis == 0) _coffset = std::min(_outc + _coffset, channels);
                // if(axis == 1) _hoffset = std::min(_outh + _hoffset, h);
                // if(axis == 2) _woffset = std::min(_outw + _woffset, w);
                }else{
                fprintf(stderr, "Horizontal split value invalid...\n");
                return -100;
                }
            }
        return 0;
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

            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _outc > 0 && _coffset % 8 == 0 && out_elempack == 8)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc * 2 / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 16u)
                        crop_pack8_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack8_neon(m, borderm, _hoffset, _woffset);

                }

                bottom_blob_sliced = bottom_blob.channel_range((_coffset + channels) / out_elempack, _outc / out_elempack);   
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q + _outc);
                    if (elemsize == 16u)
                        crop_pack8_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack8_neon(m, borderm, _hoffset, _woffset);
                }
                if(axis == 0) _coffset = std::min(_outc + _coffset, channels);
            }else{
                return -100;
            }
        }
        return 0;
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
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
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

            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _outc > 0 && _outd > 0 && _coffset % 8 == 0 && out_elempack == 8)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);
                         if (elemsize == 16u)
                            crop_pack8_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                        else
                            crop_pack8_neon(m, borderm, _hoffset, _woffset);
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
        return 0;
    }  
}

    if (elempack == 4)
    {

        if (dims == 1)
        {

        for (int i = 0; i < sranks.w; i++)
        {
            _outw = std::min(pspoints[i], w - pstarts[i]);       
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack; 

            if (_outw > 0 && _woffset % 4 == 0 && out_elempack == 4)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elemsize == 8u)
                    crop_pack4_bf16_fp16s_neon(bottom_blob, top_blob, 0, pstarts[i] / elempack);
                else
                    crop_pack4_neon(bottom_blob, top_blob, 0, pstarts[i] / elempack);

                //  _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
        return 0;
        }

        if (dims == 2)
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
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;
            if (_outw > 0 && _outh > 0 && _hoffset % 4 == 0 && out_elempack == 4)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elemsize == 8u)
                    crop_pack4_bf16_fp16s_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack4_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                if(axis == 0) _hoffset = std::min(_outh + _hoffset, h);
                if(axis == 1) _woffset = std::min(_outw + _woffset, w);
            }else{
                return -100;
            }
        }
        return 0;
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
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
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
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _outc > 0 && _coffset % 4 == 0 && out_elempack == 4)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);  
                // fprintf(stderr," W %d, H %d, C %d, created.",_outw, _outh, _outc);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
                    if (elemsize == 8u)
                        crop_pack4_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack4_neon(m, borderm, _hoffset, _woffset);

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
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw > 0 && _outh > 0 && _outc > 0 && _coffset % 4 == 0 && out_elempack == 4)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outc * 2  / out_elempack,  out_elemsize, out_elempack, opt.blob_allocator); 
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
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
                    if (elemsize == 8u)
                        crop_pack4_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack4_neon(m, borderm, _hoffset, _woffset);

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
                bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);   
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
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;
//       fprintf(stderr, "_outd %d, doffset %d, _outc %d, _coffset  %d, _outh %d, _hoffset:%d, _outw: %d, _woffset: %d elemsize:%d \n", _outd, _doffset,_outc, _coffset, _outh, _hoffset, _outw, _woffset, elemsize);


            if (_outw > 0 && _outh > 0 && _outc > 0 && _outd > 0 && _coffset % 4 == 0 && out_elempack == 4)
            {
                ncnn::Mat& top_blob = top_blobs[i];
                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < _outc; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);

                        if (elemsize == 8u)
                            crop_pack4_bf16_fp16s_neon(m, borderm, _hoffset, _woffset);
                        else
                            crop_pack4_neon(m, borderm, _hoffset, _woffset);
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
    }
#endif // __ARM_NEON

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat reference_blob_unpacked = reference_blob;
    if (ref_elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(reference_blob, reference_blob_unpacked, 1, opt_pack1);
    }

    std::vector<Mat> bottom_blobs_unpacked(2);
    bottom_blobs_unpacked[0] = bottom_blob_unpacked;
    bottom_blobs_unpacked[1] = reference_blob_unpacked;

    return HSplit::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
