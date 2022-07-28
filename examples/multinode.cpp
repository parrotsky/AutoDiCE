#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <vector>
#include <mpi.h>

static int load_labels(std::string path, std::vector<std::string>& labels)
{    
    FILE* fp = fopen(path.c_str(), "r");
     
    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);  
        std::string str_s(str);
     
        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}    
//static int print_topk(const std::vector<float>& cls_scores, int topk)
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float>& score_result)
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
        index_result.push_back(index);
        score_result.push_back(score);
    }

    return 0;
}

static int multi_classify(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
//int irank = MPI::COMM_WORLD.Get_rank();
int irank; MPI_Comm_rank(MPI_COMM_WORLD, &irank); 
MPI_Request requests[2];
MPI_Status status[2];

if(irank==0){
    ncnn::Net tx01conv1_2;
    tx01conv1_2.load_param("tx01conv1_2.param");
    tx01conv1_2.load_model("tx01conv1_2.bin");

    ncnn::Extractor exconv1_2 = tx01conv1_2.create_extractor();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);
    exconv1_2.input("data_0", in);

    ncnn::Mat conv1_2;

    exconv1_2.extract("conv1_2", conv1_2);
    MPI_Isend((float* )conv1_2, conv1_2.total(), MPI_FLOAT, 1, 
        0, MPI_COMM_WORLD, &requests[0]);

    MPI_Wait(&requests[0], &status[0]);
 }

if(irank==1){
    ncnn::Mat conv1_2(54, 54, 96);

    MPI_Irecv((float* )conv1_2, conv1_2.total(), MPI_FLOAT, 0, 
        0, MPI_COMM_WORLD, &requests[1]);

    ncnn::Net tx02prob_1;
    tx02prob_1.load_param("tx02prob_1.param");
    tx02prob_1.load_model("tx02prob_1.bin");

    ncnn::Extractor exprob_1 = tx02prob_1.create_extractor();

    MPI_Wait(&requests[1], &status[1]);
    exprob_1.input("conv1_2", conv1_2);
    ncnn::Mat prob_1;

    exprob_1.extract("prob_1", prob_1);
    cls_scores.resize(prob_1.w);

    for (int j = 0; j < prob_1.w; j++)
    {
        cls_scores[j] = prob_1[j];
    }

    std::vector<std::string> labels;
    load_labels("synset_words.txt", labels);
    std::vector<int> index;
    std::vector<float> score;
    print_topk(cls_scores, 3, index, score);
    for (int i = 0; i < index.size(); i++)
    {
        fprintf(stderr, "%s \n", labels[index[i]].c_str());
    }
 }

return 0;

}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    //world_size = MPI::COMM_WORLD.Get_size();

    // Get the rank of the process
    int world_rank;
    //world_rank = MPI::COMM_WORLD.Get_rank();
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


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
    MPI_Finalize();

    return 0;
}
            
