#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>
#include <mpi.h>

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

static int multi_classify(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
int irank = MPI::COMM_WORLD.Get_rank();
MPI_Request requests[6];
MPI_Status status[6];

if(irank==0){
    ncnn::Net tx01conv1_2;
    tx01conv1_2.load_param("tx01conv1_2.param");
    tx01conv1_2.load_model("tx01conv1_2.bin");

    ncnn::Extractor exconv1_2 = tx01conv1_2.create_extractor();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    exconv1_2.input("data_0", in);

    ncnn::Mat conv1_2;

    exconv1_2.extract("conv1_2", conv1_2);
    MPI_Isend((float* )conv1_2, conv1_2.total(), MPI_FLOAT, 2, 
        0, MPI_COMM_WORLD, &requests[0]);

    MPI_Wait(&requests[0], &status[0]);
 }

if(irank==1){
    ncnn::Mat conv2_2(26, 26, 256);

    MPI_Irecv((float* )conv2_2, conv2_2.total(), MPI_FLOAT, 2, 
        2, MPI_COMM_WORLD, &requests[5]);

    ncnn::Net tx02norm2_1;
    tx02norm2_1.load_param("tx02norm2_1.param");
    tx02norm2_1.load_model("tx02norm2_1.bin");

    ncnn::Extractor exnorm2_1 = tx02norm2_1.create_extractor();

    MPI_Wait(&requests[5], &status[5]);
    exnorm2_1.input("conv2_2", conv2_2);
    ncnn::Mat norm2_1;

    exnorm2_1.extract("norm2_1", norm2_1);
    MPI_Isend((float* )norm2_1, norm2_1.total(), MPI_FLOAT, 2, 
        1, MPI_COMM_WORLD, &requests[1]);

    MPI_Wait(&requests[1], &status[1]);
 }

if(irank==2){
    ncnn::Mat norm2_1(26, 26, 256);

    MPI_Irecv((float* )norm2_1, norm2_1.total(), MPI_FLOAT, 1, 
        1, MPI_COMM_WORLD, &requests[4]);

    ncnn::Mat conv1_2(54, 54, 96);

    MPI_Irecv((float* )conv1_2, conv1_2.total(), MPI_FLOAT, 0, 
        0, MPI_COMM_WORLD, &requests[3]);

    ncnn::Net nx01conv2_2;
    nx01conv2_2.load_param("nx01conv2_2.param");
    nx01conv2_2.load_model("nx01conv2_2.bin");

    ncnn::Extractor exconv2_2 = nx01conv2_2.create_extractor();

    MPI_Wait(&requests[3], &status[3]);
    exconv2_2.input("conv1_2", conv1_2);
    ncnn::Mat conv2_2;

    exconv2_2.extract("conv2_2", conv2_2);
    MPI_Isend((float* )conv2_2, conv2_2.total(), MPI_FLOAT, 1, 
        2, MPI_COMM_WORLD, &requests[2]);

    MPI_Wait(&requests[2], &status[2]);
    ncnn::Net nx01prob_1;
    nx01prob_1.load_param("nx01prob_1.param");
    nx01prob_1.load_model("nx01prob_1.bin");

    ncnn::Extractor exprob_1 = nx01prob_1.create_extractor();

    MPI_Wait(&requests[4], &status[4]);
    exprob_1.input("norm2_1", norm2_1);
    ncnn::Mat prob_1;

    exprob_1.extract("prob_1", prob_1);
    cls_scores.resize(prob_1.w);

    for (int j = 0; j < prob_1.w; j++)
    {
        cls_scores[j] = prob_1[j];
    }

    print_topk(cls_scores, 3);

 }

return 0;

}

int main(int argc, char** argv)
{
    MPI::Init(argc, argv);

    // Get the number of processes
    int world_size;
    world_size = MPI::COMM_WORLD.Get_size();

    // Get the rank of the process
    int world_rank;
    world_rank = MPI::COMM_WORLD.Get_rank();

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
    multi_classify(m, cls_scores);

    // Finalize the MPI environment.
    MPI::Finalize();

    return 0;
}
            
