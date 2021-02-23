#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

bool Comp(const DMatch& a, const DMatch& b) {
	return a.distance > b.distance;
}

int main(int, char**) {
    cv::Mat img1 = cv::imread("../1.png");
    cv::Mat img2 = cv::imread("../2.png");
    cv::Mat img11,img22;
    cv::cvtColor(img1,img11,cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2,img22,cv::COLOR_BGR2GRAY);
    //cv::imshow("diao",img1);
    std::vector<KeyPoint> keypoints_1,keypoints_2;
    auto orb = cv::ORB::create();
    cv::Mat img_desc1,img_desc2;
    cv::Mat mask_mat;
    orb->detectAndCompute(img11,cv::Mat(),keypoints_1,img_desc1);
    orb->detectAndCompute(img22,cv::Mat(),keypoints_2,img_desc2);
    auto bf = cv::BFMatcher::create();
    std::vector<DMatch> matches;
    bf->match(img_desc1,img_desc2,matches);
    sort(matches.begin(),matches.end(),Comp);
    std::vector<DMatch> matches2(matches.begin(),matches.begin()+30);
    cv::Mat output_img;
    cv::drawMatches(img11,keypoints_1,img22,keypoints_2,matches2,output_img);
    cv::imshow("fuck",output_img);
    std::cout<<matches2.size()<<std::endl;
    cv::waitKey(0);
}
