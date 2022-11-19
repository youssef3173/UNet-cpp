#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <map>
#include <vector>

#define IMAGE_SIZE_X 128
#define IMAGE_SIZE_Y 256
#define CHANNELS 3
#define CLASSES 8

using namespace cv;
using namespace std;


int main(int argc, const char **argv){
    auto t1 = std::chrono::high_resolution_clock::now();

    map<int,vector<int>> colors;
    colors[0] = { 0, 0, 0};
    colors[1] = { 128, 64, 128};
    colors[2] = { 70, 70, 70};
    colors[3] = { 153, 153, 153};
    colors[4] = { 107, 142, 35};
    colors[5] = { 70, 130, 180};
    colors[6] = { 220, 20, 60};
    colors[7] = { 0, 0, 142};

    /*  // Print a Map:
    map<int,vector<int>>::iterator it;
    for (it = colors.begin(); it != colors.end(); ++it) {
        cout << "key: " << it->first << "  :: "
             << "3 values: " << it->second.at(0) << " , " << it->second.at(1) << " , " << it->second.at(2) << endl;
    }
    */


    if (argc != 4) {
        std::cerr << "Usage: UNet <path-to-exported-script-module> "
        << "<path-to-input-image>" << "<path-to-save-result>"
        << std::endl;
        return -1;
    }

    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Load the model: AlexNet:
    torch::jit::script::Module model = torch::jit::load( argv[1]);

    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Load Image:
    cv::Mat image = cv::imread( argv[2]);  // CV_8UC3
    if (image.empty() || !image.data) {
        cout << "Can't load or open the image" << endl;
        return -1;
    }

    cv::cvtColor( image, image, cv::COLOR_BGR2RGB);  // GRAY);

    // scale image to fit:
    cv::Size scale( IMAGE_SIZE_Y, IMAGE_SIZE_X);
    cv::resize( image, image, scale);

    // convert [unsigned int] to [float]
    image.convertTo( image, CV_32FC3, 1.0f / 255.0f);


    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Inference phase:
    try{
        // Image to Tensor:
        auto input_tensor = torch::from_blob( image.data, { 1, IMAGE_SIZE_X, IMAGE_SIZE_Y, CHANNELS});     // , torch::kFloat32);
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        // Forward Pass Through the Model:
        input_tensor = input_tensor.contiguous().view( {-1, CHANNELS, IMAGE_SIZE_X, IMAGE_SIZE_Y} );
        torch::Tensor out_tensor = model.forward( {input_tensor} ).toTensor();



        // Convert Tensor of Labels to Image:
        cv::Mat out_mat = cv::Mat::zeros( {IMAGE_SIZE_Y, IMAGE_SIZE_X}, CV_32FC3);
        torch::Tensor tmp = torch::randn( out_tensor.sizes()[1] );

        int idx;
        for( int x = 0; x < out_tensor.sizes()[2]; x++){
            for( int y = 0; y < out_tensor.sizes()[3]; y++){
                for( int n = 0; n < out_tensor.sizes()[1]; n++ ){
                    tmp[n] = out_tensor[0][n][x][y].item<float>();
                }

                idx = torch::argmax( tmp).item<int>();
                for (int c = 0; c < 3; c++){
                    out_mat.at<float>( x, 3*y+c) = colors[idx].at(c);
                }
            }
        }

        // Save Image:
        out_mat.convertTo( out_mat, CV_8U);

        cv::Size scale( 4*IMAGE_SIZE_Y, 4*IMAGE_SIZE_X);
        cv::resize( out_mat, out_mat, scale);

        cv::imwrite( argv[3], out_mat);
        cv::waitKey(0);


        auto t2 = std::chrono::high_resolution_clock::now();
        cout << "Done Successfully in: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
    }
    catch (const c10::Error& e) {
        std::cerr << " ecountered error in the inference phase \n";
        return -1;
    }



    return 0;
}