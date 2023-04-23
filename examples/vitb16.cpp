#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "net.h"
#if 1
void pretty_print(const ncnn::Mat& m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}
#endif
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);

    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    // cv::Mat img = cv::imread("image.ppm", CV_LOAD_IMAGE_GRAYSCALE);
    int w = img.cols;
    int h = img.rows;
    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_RGB, w, h, 224, 224);
    float mean[1] = {128.f};
    float norm[1] = {1 / 128.f};
    in.substract_mean_normalize(mean, norm);
    ncnn::Net net;
    net.load_param("vit_b_16.ncnn.param");
    net.load_model("vit_b_16.ncnn.bin");
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("in0", in);
    // pretty_print(in);
    std::vector<const char*> blobs = net.output_names();
    ncnn::Mat feat;
    ex.extract("out0", feat);
    pretty_print(feat);
    return 0;
}