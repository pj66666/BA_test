#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>

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

// SE3转换为OpenGL中的4x4变换矩阵
pangolin::OpenGlMatrix SE3ToOpenGlMat(const Eigen::Matrix3d& G_R_C, const Eigen::Vector3d& G_p_C) {
    pangolin::OpenGlMatrix p_mat;
    
    p_mat.m[0] = G_R_C(0, 0);
    p_mat.m[1] = G_R_C(1, 0);
    p_mat.m[2] = G_R_C(2, 0);
    p_mat.m[3] = 0.;

    p_mat.m[4] = G_R_C(0, 1);
    p_mat.m[5] = G_R_C(1, 1);
    p_mat.m[6] = G_R_C(2, 1);
    p_mat.m[7] = 0.;

    p_mat.m[8] = G_R_C(0, 2);
    p_mat.m[9] = G_R_C(1, 2);
    p_mat.m[10] = G_R_C(2, 2);
    p_mat.m[11] = 0.;

    p_mat.m[12] = G_p_C(0);
    p_mat.m[13] = G_p_C(1);
    p_mat.m[14] = G_p_C(2);
    p_mat.m[15] = 1.;

    return p_mat;
}

// 在OpenGL中绘制一个表示相机的简单模型
void DrawOneCamera(const Eigen::Matrix3d& G_R_C, const Eigen::Vector3d& G_p_C) {
    const float w = 1.5;   // 在OpenGL中绘制一个表示相机的简单模型
    const float h = w * 0.75;           // 相机的高度
    const float z = w * 0.6;            // 

    // 使用旋转矩阵 G_R_C 和平移向量 G_p_C 创建一个 OpenGL 变换矩阵 G_T_C
    pangolin::OpenGlMatrix G_T_C = SE3ToOpenGlMat(G_R_C, G_p_C);

    // 保存当前的模型视图矩阵
    glPushMatrix();

    #ifdef HAVE_GLES
    // 如果是 OpenGL ES，使用 glMultMatrixf 将 G_T_C 应用到当前模型视图矩阵中
    glMultMatrixf(G_T_C.m);
    #else
    // 如果是标准的 OpenGL，则使用 glMultMatrixd 将 G_T_C 应用到当前模型视图矩阵中
    glMultMatrixd(G_T_C.m);
    #endif

    // 设置绘制线段的宽度
    glLineWidth(3);

    // 开始绘制相机模型的线段
    glBegin(GL_LINES);
    // 绘制相机的四个边线
    glVertex3f(0, 0, 0);         // 相机原点
    glVertex3f(w, h, z);         // 两个顶点构成了相机模型的一条边线，从相机的原点 (0, 0, 0) 到右上角的顶点 (w, h, z)
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);        // 从相机的原点 (0, 0, 0) 到右下角的顶点 (w, -h, z)
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    // 绘制相机的四条边框线
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    // 恢复之前保存的模型视图矩阵
    glPopMatrix();

    // 在相机附近绘制一个坐标轴
    pangolin::glDrawAxis(G_T_C, 0.2);

}

// 画出所有相机的位姿
void DrawCameras(vector<Frame> &cameras) {
    for(auto cam : cameras){
        DrawOneCamera(cam.Rwc, cam.twc);
    }
}

void DrawFeatures(vector<Eigen::Vector3d> points) {
    glPointSize(5);
    glBegin(GL_POINTS);

    for(Eigen::Vector3d& pt : points) {
        glVertex3f(pt[0], pt[1], pt[2]);
    }

    glEnd();
}

void VisualizerRun(vector<Frame> &cameras, vector<Eigen::Vector3d> points) {
    // 设置窗口大小和名称
    pangolin::CreateWindowAndBind("simdata-visual", 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 创建一个菜单，里面会提供相应的按钮设置
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));

    pangolin::Var<bool> menu_follow_cam("menu.Follow Camera", true, true);
    pangolin::Var<int> grid_scale("menu.Grid Size (m)", 100, 1, 500);
    pangolin::Var<bool> show_grid("menu.Show Grid", true, true);
    pangolin::Var<bool> show_map("menu.Show Map", true, true);
    pangolin::Var<bool> show_cam("menu.Show Camera", true, true);
    pangolin::Var<bool> show_traj("menu.Show Traj", true, true);
    pangolin::Var<bool> show_gt_traj("menu.Show GroundTruth", true, true);
    pangolin::Var<bool> show_raw_odom("menu.Show Raw Odom", true, true);
    pangolin::Var<bool> show_gps_point("menu.Show GPS", true, true);

    // Define Camera Render Object (for view / scene browsing)
    // 场景在窗口中的呈现方式和相机的视角
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1920, 1080, 500, 500, 960, 540, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0));
                      
    // Add named OpenGL viewport to window and provide 3D Handler 
    // 进行鼠标和键盘交互
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1920.0f/1080.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix G_T_C;
    G_T_C.SetIdentity();

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f,1.0f);

        // Draw grid.
        if (show_grid.Get()) {
            glColor3f(0.3f, 0.3f, 0.3f);
            pangolin::glDraw_z0(grid_scale, 1000);
        }

        // Draw camera poses.
        if (show_cam.Get()) {
            glColor3f(0.0f, 1.0f, 0.0f);
            DrawCameras(cameras);
        }

        // Draw map points.
        if (show_map.Get()) {
            glColor3f(0.0f, 0.0f, 1.0f);
            DrawFeatures(points);
        }
        pangolin::FinishFrame();
    }
}



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
    VisualizerRun(cameras, points);
 
    return 0;
}

