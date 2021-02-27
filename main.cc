#include <iostream>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace cv;
using namespace std;


const double cx = 325.5;
const double cy = 253.5;
const double fx = 518.0;
const double fy = 519.0;


bool match_keypoint(const cv::Mat& img1,const cv::Mat& img2,vector<cv::Point2f>& points1, vector<cv::Point2f>& points2,int point_num=30,bool plot = true){

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
    if(matches.size()<point_num)
    {
        return false;
    }
    sort(matches.begin(),matches.end());

    for(auto pt = matches.begin();pt!=matches.begin()+point_num;pt++)
    {
        points1.push_back(keypoints_1[pt->queryIdx].pt);
        points2.push_back(keypoints_2[pt->trainIdx].pt);
    }
    
    if (plot) {
        Mat output_img;
        std::vector<DMatch> matches2(matches.begin(),matches.begin()+point_num);
        cv::drawMatches(img11,keypoints_1,img22,keypoints_2,matches2,output_img, 
                            Scalar_<double>::all(-1), Scalar_<double>::all(-1),std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("fuck",output_img);
        for(const auto& mm:matches2)
        {
            std::cout<<mm.distance<<std::endl;
        }
        cv::waitKey(0);
    }
    return true;
}


int main(int, char**) {
    cv::Mat img1 = cv::imread("../1.png");
    cv::Mat img2 = cv::imread("../2.png");
    vector<cv::Point2f> points1,points2;
    match_keypoint(img1,img2,points1,points2,30,true);




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

    // 添加节点
    // 两个位姿节点
    g2o::VertexSE3Expmap* pst1 = new g2o::VertexSE3Expmap();
    pst1->setId(0);
    pst1->setFixed(true); // 第一个点固定为零
    // 预设值为单位Pose，因为我们不知道任何信息
    pst1->setEstimate(g2o::SE3Quat());

    g2o::VertexSE3Expmap* pst2 = new g2o::VertexSE3Expmap();
    pst2->setId(1);
    // 预设值为单位Pose，因为我们不知道任何信息
    pst2->setEstimate(g2o::SE3Quat());

    optimizer.addVertex(pst1);
    optimizer.addVertex(pst2);


    // 很多个特征点的节点
    // 以第一帧为准
    for ( size_t i=0; i<points1.size(); i++ )
    {
        g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
        v->setId( 2 + i );
        // 由于深度不知道，只能把深度设置为1了
        double z = 1;
        double x = ( points1[i].x - cx ) * z / fx; 
        double y = ( points1[i].y - cy ) * z / fy; 
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }

    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    // 准备边
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<points1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(points1[i].x, points1[i].y ) );
         edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    // 第二帧
    for ( size_t i=0; i<points2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(points2[i].x, points2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
}
