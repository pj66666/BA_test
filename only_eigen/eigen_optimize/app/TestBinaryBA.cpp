#include <iostream>
#include <random>

#include "backend/vertex_pose.h"
#include "backend/vertex_point_xyz.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qcw(R.transpose()), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qcw;
    Eigen::Vector3d twc;

    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征以及特征id
};

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points) {
    int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;     // 相机数目

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            // Pc = Rcw*(Pw - twc) = Rcw*Pw - Rcw*twc = Rcw*Pw + tcw = Pc
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}

int main() {
    // 准备数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points);     // 生成3个姿态、20个路标点
    Eigen::Quaterniond qic(1, 0, 0, 0);     // 表示了相机到imu的旋转，这里简化了
    Eigen::Vector3d tic(0, 0, 0);

    // 1. 构建 problem
    Problem problem(Problem::ProblemType::SLAM_PROBLEM);

    // 2.创建所有 Pose顶点
    std::vector<std::shared_ptr<VertexPose> > vertexCams_vec;
    for (size_t i = 0; i < cameras.size(); ++i) {
        std::shared_ptr<VertexPose> vertexCam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qcw.x(), cameras[i].qcw.y(), cameras[i].qcw.z(), cameras[i].qcw.w(); //平移和四元数
        vertexCam->SetParameters(pose); // 优化参数变量

            //    if(i < 2)
            //        vertexCam->SetFixed();

        problem.AddVertex(vertexCam);
        vertexCams_vec.push_back(vertexCam);
    }

    // 3.创建所有 landmark顶点
    vector<shared_ptr<VertexPointXYZ>> allPoints;
    for (size_t i = 0; i < points.size(); ++i) {
        //假设所有特征点的起始帧为第0帧， 逆深度容易得到
        Eigen::Vector3d Pw = points[i];
 
        // 初始化特征 vertex
        shared_ptr<VertexPointXYZ> verterxPoint(new VertexPointXYZ());
        verterxPoint->SetParameters(Pw);     // 优化三维点
        problem.AddVertex(verterxPoint);
        allPoints.push_back(verterxPoint);

        // 4.每个特征对应的投影误差, 第 0 帧为起始帧
        for (size_t j = 1; j < cameras.size(); ++j) {

            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
            // 纯视觉SLAM里面一般都是二元边，顶点xyz和位姿
            shared_ptr<EdgeReprojectionXYZ> edge(new EdgeReprojectionXYZ(pt_j));

            std::vector<std::shared_ptr<Vertex> > edge_vertex;
            // 依次添加误差边对应的顶点--XYZ、Tcw
            edge_vertex.push_back(verterxPoint);
            edge_vertex.push_back(vertexCams_vec[j]);
            edge->SetVertex(edge_vertex);

            problem.AddEdge(edge);
        }
    }

    problem.Solve(5); //一共迭代5次

    std::cout<<"------------ pose translation ----------------"<<std::endl;
    for (size_t i = 0; i < vertexCams_vec.size(); ++i) {
        std::cout<<"translation after opt: "<< i <<" :"<< vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: "<<cameras[i].twc.transpose()<<std::endl;
    }


    return 0;
}