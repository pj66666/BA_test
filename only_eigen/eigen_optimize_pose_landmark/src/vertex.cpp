#include "myslam/vertex.h"
#include <iostream>

namespace myslam {


unsigned long global_vertex_id = 0;

Vertex::Vertex(int num_dimension, int local_dimension) {
    parameters_.resize(num_dimension, 1);
    local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
    id_ = global_vertex_id++;
}

Vertex::~Vertex() {}

int Vertex::Dimension() const {
    return parameters_.rows();
}

int Vertex::LocalDimension() const {
    return local_dimension_;
}

void Vertex::Plus(const VecX &delta) {
    parameters_ += delta;
}

std::string VertexPose_PJ::TypeInfo() const  { 
        return "VertexPose"; 
}

std::string VertexPointXYZ_PJ::TypeInfo() const { 
    return "VertexPointXYZ"; 
} 


/*******************位姿增量复写************************/
void VertexPose_PJ::Plus(const VecX &delta) {
    VecX &parameters = Parameters();
    parameters.head<3>() += delta.head<3>();
    Eigen::Quaterniond q(parameters[6], parameters[3], parameters[4], parameters[5]);
    q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  // right multiplication with so3
    q.normalized();
    parameters[3] = q.x();
    parameters[4] = q.y();
    parameters[5] = q.z();
    parameters[6] = q.w();
//    Qd test = Sophus::SO3d::exp(Vec3(0.2, 0.1, 0.1)).unit_quaternion() * Sophus::SO3d::exp(-Vec3(0.2, 0.1, 0.1)).unit_quaternion();
//    std::cout << test.x()<<" "<< test.y()<<" "<<test.z()<<" "<<test.w() <<std::endl;
}


}