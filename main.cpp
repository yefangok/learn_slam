#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int, char**) {
    cv::Mat img1 = cv::imread("../1.png");
    cv::Mat img2 = cv::imread("../2.png");
    //cv::imshow("diao",img1);
    std::vector<KeyPoint> keypoints_1,keypoints_2;
    auto orb = cv::ORB::create();
    cv::Mat img_desc1,img_desc2;
    cv::Mat mask_mat;
    orb->detectAndCompute(img1,cv::Mat(),keypoints_1,img_desc1);
    std::cout << keypoints_1.size() << std::endl;
    std::cout << img_desc1.size() << std::endl;
    cv::imshow("fuck",img1);
}
