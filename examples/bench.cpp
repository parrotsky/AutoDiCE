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

#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>
#include "benchmark.h"

//#ifdef _WIN32
//#include <algorithm>
//#include <windows.h> // Sleep()
//#else
#include <unistd.h> // sleep()
//#endif
#include "cpu.h"
#include "gpu.h"

static int detect_resnet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net resnet;
        int num_threads = ncnn::get_cpu_count();



    //resnet.opt.use_vulkan_compute = true;
    resnet.opt.use_vulkan_compute = false;
    resnet.opt.num_threads = num_threads;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    //resnet.load_param("resnet50-v2-7.param");
    //resnet.load_model("resnet50-v2-7.bin");

    
    resnet.load_param("bvlcalexnet-9.param");
    resnet.load_model("bvlcalexnet-9.bin");

    //resnet.load_param("resnet152-v2-7.param");
    //resnet.load_model("resnet152-v2-7.bin");
    //resnetv27_dense0_fwd
                                         
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
                                         
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);
                                         
//const float std_vals[3] = {57.0f, 57.0f, 58.0f};

    //const float std_vals[3] = {0.225f, 0.224f,0.229f};
    //in.substract_mean_normalize(mean_vals, std_vals);
    //
    ncnn::Mat out;
    ncnn::Extractor ex = resnet.create_extractor();
    ex.input("data_0", in);
    ex.extract("prob_1", out);  ////resnet101
//ex.extract("resnetv17_dense0_fwd", out);
    // ex.extract("resnetv24_dense0_fwd", out);  ////resnet50
    //ex.extract("resnetv27_dense0_fwd", out);     ////resnet152
#if NCNN_BENCHMARK
    int g_warmup_loop_count = 5;
    int g_loop_count = 10;
    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
	    ex = resnet.create_extractor();
    ex.input("data_0", in);
    ex.extract("prob_1", out);  ////resnet101
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ex = resnet.create_extractor();
    ex.input("data_0", in);
    ex.extract("prob_1", out);  ////resnet101
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "Resnet-50V2  min = %7.2f  max = %7.2f  avg = %7.2f\n",  time_min, time_max, time_avg);

#endif

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_resnet(m, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}
