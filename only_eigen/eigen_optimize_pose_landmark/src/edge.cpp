#include "myslam/vertex.h"
#include "myslam/edge.h"
//#include <glog/logging.h>
#include <iostream>

using namespace std;

namespace myslam {


unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string> &verticies_types) {
    residual_.resize(residual_dimension, 1);
//    verticies_.resize(num_verticies);      // TODO:: 这里可能会存在问题，比如这里resize了3个空,后续调用edge->addVertex. 使得vertex前面会存在空元素
    if (!verticies_types.empty())
        verticies_types_ = verticies_types;
    jacobians_.resize(num_verticies);
    id_ = global_edge_id++;

    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;

//    cout<<"Edge construct residual_dimension="<<residual_dimension
//            << ", num_verticies="<<num_verticies<<", id_="<<id_<<endl;
}

Edge::~Edge() {}

double Edge::Chi2() {
    // TODO::  we should not Multiply information here, because we have computed Jacobian = sqrt_info * Jacobian
    return residual_.transpose() * information_ * residual_;
//    return residual_.transpose() * residual_;   // 当计算 residual 的时候已经乘以了 sqrt_info, 这里不要再乘
}

bool Edge::CheckValid() {
    if (!verticies_types_.empty()) {
        // check type info
        for (size_t i = 0; i < verticies_.size(); ++i) {
            if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
                cout << "Vertex type does not match, should be " << verticies_types_[i] <<
                     ", but set to " << verticies_[i]->TypeInfo() << endl;
                return false;
            }
        }
    }
/*
    CHECK_EQ(information_.rows(), information_.cols());
    CHECK_EQ(residual_.rows(), information_.rows());
    CHECK_EQ(residual_.rows(), observation_.rows());

    // check jacobians
    for (size_t i = 0; i < jacobians_.size(); ++i) {
        CHECK_EQ(jacobians_[i].rows(), residual_.rows());
        CHECK_EQ(jacobians_[i].cols(), verticies_[i]->LocalDimension());
    }
    */
    return true;
}


/**************************************************************/

void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Eigen::Quaterniond Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_camera_i = Qi * pts_w + Pi;   // Pc = Rcw*Pw + tcw

    double dep_i = pts_camera_i.z();
    
    residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
}


void EdgeReprojectionXYZ::ComputeJacobians() {

    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_camera_i = Qi * pts_w + Pi;    // Rcw*Pw + tcw

    double dep_i = pts_camera_i.z();

    Mat33 Ri = Qi.toRotationMatrix();   // Rcw
    
    Mat23 reduce(2, 3);
    reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
        0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() =  Eigen::Matrix3d::Identity();   // I
    jaco_i.rightCols<3>() =  -Sophus::SO3d::hat(pts_camera_i); // pts_imu_i反对称矩阵
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 3> jacobian_feature;
    jacobian_feature = reduce * Ri;

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;

}


}