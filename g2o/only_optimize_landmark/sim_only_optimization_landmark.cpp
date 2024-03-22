#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <sophus/se3.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征以及特征id
};

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points, vector<Eigen::Vector3d> &points_with_noise) {
    int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 2;     // 相机数目

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
        std::uniform_real_distribution<double> xy_rand(0.0, 4.0); 
        std::uniform_real_distribution<double> z_rand(4., 8.);


        Eigen::Vector3d Pw(xy_rand(generator),xy_rand(generator), z_rand(generator ));
        points.push_back(Pw);
        
        // 给三维点加噪声---智障错误，带噪声的数据也是先随机生成了。。。。我就说
        Eigen::Vector3d Pw_with_noise(Pw[0] + noise_pdf(generator),
                           Pw[1] + noise_pdf(generator), 
                           Pw[2] + noise_pdf(generator));
        points_with_noise.push_back(Pw_with_noise);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            // Pc = Rcw*(Pw - twc) = Rcw*Pw - Rcw*twc = Rcw*Pw + tcw = Pc
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();  // 归一化图像平面
            // 这里认为观测没有噪声，观测和位姿都是正确的
            // Pc[0] += noise_pdf(generator);
            // Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}


class Vertexonlylandmark: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
    Vertexonlylandmark(){}
    // 重置
    virtual void setToOriginImpl() override
    {
        _estimate.fill(0.);		// 设置初始值
    }

    // update 数组指针？
    virtual void oplusImpl(const double *update) override
    {
        // 要加上const
        Eigen::Map<const Eigen::Vector3d> up(update);
    	_estimate += up;   // 路标点的更新量直接加在原值上面即可
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};


class Edgeonlylandmark : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, Vertexonlylandmark> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Edgeonlylandmark(Sophus::SE3d &T) :_T(T) {}

  // 1 计算3D-2D投影误差
  virtual void computeError() override {
    const Vertexonlylandmark *v = static_cast<Vertexonlylandmark*> (_vertices[0]);
      
    Eigen::Vector3d XYZ = v->estimate();     // return _estimate
      
    Eigen::Vector3d pos_pixel =  _T * XYZ;
      
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();

    // cout << "Error: " << _error.norm() << endl; // 打印误差值
  }

    
  // 2 计算3D-2D投影误差_error的对landmark雅可比 
  virtual void linearizeOplus() override {
    const Vertexonlylandmark *v = static_cast<Vertexonlylandmark*> (_vertices[0]);
    Eigen::Vector3d XYZ = v->estimate();
      
    Eigen::Vector3d pos_cam = _T * XYZ;	// 投影
    

    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;


    Eigen::Matrix<double, 2, 3> jacobianuv;
    jacobianuv << 1/Z, 0, -X/Z2, 0, 1/Z, -Y/Z2;     // -Y/Z2之前没加上负号！
    
    // 注意这里是观测-预测，要有一个负号 Rwc x Rcw √
    _jacobianOplusXi = -jacobianuv * _T.rotationMatrix().transpose();
  }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

private:
  Sophus::SE3d _T;

};


int main() {
    // 1 准备数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points, points_with_noise;
    GetSimDataInWordFrame(cameras, points, points_with_noise);     // 生成3个姿态、20个路标点
    // for(int k = 0; k < cameras.size(); k++){
    //     std::cout << cameras[k].Rwc << std::endl;
    // }
    // for(int j = 0; j < points.size(); j++){
    //     cout << "实际点:" << points[j].transpose() << "带噪声点:" << points_with_noise[j].transpose() << endl;
    // }

    // 2 设置求解器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型

    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // 3 添加XYZ顶点 
    int mnid = 0;
    std::vector<Vertexonlylandmark*> allvPoints(points_with_noise.size());
    // std::vector<shared_ptr<Vertexonlylandmark>> allvPoints_share;
    for(size_t i = 0; i < points_with_noise.size(); i++){
        Eigen::Vector3d l3d = points_with_noise[i];
        Vertexonlylandmark* vPoint = new Vertexonlylandmark();
        // std::shared_ptr<Vertexonlylandmark> vPoint = std::make_shared<Vertexonlylandmark>();  貌似g2o这个版本并不支持

        vPoint->setEstimate(l3d);       // 顶点（XYZ）测量值即带噪声的路标顶数据
        cout << "Initial estimate for point error" << i << ": " << l3d.transpose() - points[i].transpose() << endl;
        vPoint->setId(i);
        vPoint->setFixed(false);
        vPoint->setMarginalized(true); // 仿真，没必要边缘化，ORB里面设置了，认为一个点无所谓
        optimizer.addVertex(vPoint);
        // allvPoints.push_back(vPoint);   // 记录每一个顶点，后续分析误差
        allvPoints[i] = vPoint;
        // std::cout << "*******" << i << "*******" << std::endl;
        // 4 添加edge
        for(int j = 0; j < cameras.size(); j++){
            
            Sophus::SE3d SE3(cameras[j].Rwc, cameras[j].twc);   // nt了把j写成i，数组越界，对应旋转矩阵时0
            // cout << "*******" << mnid << "*******" << endl;     // 3*20=60 edges
            Edgeonlylandmark* edge = new Edgeonlylandmark(SE3);

            edge->setId(mnid); 
            edge->setVertex(0, vPoint);
            Eigen::Vector2d Measurement = cameras[j].featurePerId[i].head<2>();
            edge->setMeasurement(Measurement);
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            mnid++;
        }
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(25);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // std::cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    for(int j = 0; j < allvPoints.size(); j++){
        Vertexonlylandmark* v = dynamic_cast<Vertexonlylandmark*>(optimizer.vertex(j));
        cout << "g2o优化后点:" << v->estimate().transpose() << "实际点:" << points[j].transpose() << endl;
        // cout << "g2o优化后点:" << allvPoints[j]->estimate().transpose() << "实际点:" << points[j].transpose() << endl;
    }

    return 0;
}

