#include <iostream>
#include <random>
#include <unordered_map>
#include <memory>
#include "vertex.h"

using namespace myslam;
using namespace std;


int main() {

    // 2.创建所有 Pose顶点
    std::vector<std::shared_ptr<myslam::VertexPose>> vertexCams_vec;
    for (size_t i = 0; i < 10; ++i) {
        std::shared_ptr<myslam::VertexPose> vertexCam(new myslam::VertexPose());
        if (vertexCam) {
            cout << vertexCam->TypeInfo();
        } else {
            cout << "vertexCam is a null pointer";
        }

        Eigen::VectorXd pose(7);
        pose << 1, 1, 1, 1, 1, 1, 1; //平移和四元数
        vertexCam->SetParameters(pose); // 优化参数变量

        vertexCams_vec.push_back(vertexCam);
    }

    return 0;
}