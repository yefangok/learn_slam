#include <iostream>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

using namespace cv;
using namespace std;

void match_keypoint(){
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
    sort(matches.begin(),matches.end());
    std::vector<DMatch> matches2(matches.begin(),matches.begin()+30);
    Mat output_img;
    cv::drawMatches(img11,keypoints_1,img22,keypoints_2,matches2,output_img, 
                        Scalar_<double>::all(-1), Scalar_<double>::all(-1),std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("fuck",output_img);
    std::cout<<matches2.size()<<std::endl;
    for(const auto& mm:matches2)
    {
        std::cout<<mm.distance<<std::endl;
    }
    cv::waitKey(0);
}

int main(int, char**) {
    match_keypoint();
    g2o::SparseOptimizer optimizer;
        // 使用Cholmod中的线性方程求解器
    //g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    auto linearSolver = std::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    // 6*3 的参数
    //g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
    auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    // L-M 下降 
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);
}
