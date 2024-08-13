#ifndef FC_POINT_CLOUD_UTILS_H
#define FC_POINT_CLOUD_UTILS_H

#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <Eigen/SVD>
#include <Eigen/Dense>
//#include <opencv2/core.hpp>
#include "kdtree.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include <iostream>

bool FitPlane(std::vector<Eigen::Matrix<float, 3, 1>>& data, Eigen::Matrix<float, 4, 1>& plane_coeffs, float eps = 1e-2);
bool FitPlane(std::vector<Eigen::Matrix<double, 3, 1>>& data, Eigen::Matrix<double, 4, 1>& plane_coeffs, float eps = 1e-2);

pcl::PCLPointCloud2::Ptr fromROSmsgs(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
pcl::PCLPointCloud2::Ptr VoxelGrid(const pcl::PCLPointCloud2::Ptr cloud);
//去除打到车体的点云
pcl::PointCloud<pcl::PointXYZI>::Ptr removeCarPoints_FC(const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud_ptr,double x=4,double y=2,double z=8,double RTF=1);

inline Eigen::Vector3d ToVec3d(const pcl::PointXYZI& pt) { return pt.getVector3fMap().cast<double>(); }

template <typename S>
inline pcl::PointXYZI ToPointType(const Eigen::Matrix<S, 3, 1>& pt) {
    pcl::PointXYZI p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    return p;
}

class Icp3d {
   public:
    struct Options {
        int max_iteration_ = 20;                // 最大迭代次数
        double max_nn_distance_ = 1.0;          // 点到点最近邻查找时阈值
        double max_plane_distance_ = 0.05;      // 平面最近邻查找时阈值
        double max_line_distance_ = 0.5;        // 点线最近邻查找时阈值
        int min_effective_pts_ = 10;            // 最近邻点数阈值
        double eps_ = 1e-2;                     // 收敛判定条件
        bool use_initial_translation_ = true;  // 是否使用初始位姿中的平移估计
    };

    Icp3d() {}
    Icp3d(Options options) : options_(options) {}

    /// 设置目标的Scan
    void SetTarget(pcl::PointCloud<pcl::PointXYZI>::Ptr target) {
        target_ = target;
        BuildTargetKdTree();
        // 计算点云中心
        target_center_ = std::accumulate(target->points.begin(), target_->points.end(), Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d& c, const pcl::PointXYZI& pt) -> Eigen::Vector3d { return c + ToVec3d(pt); }) /
                         target_->size();
        //std::cout << "target center: " << target_center_.transpose();
    }

    /// 设置被配准的Scan
    void SetSource(pcl::PointCloud<pcl::PointXYZI>::Ptr source) {
        source_ = source;
        source_center_ = std::accumulate(source_->points.begin(), source_->points.end(), Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d& c, const pcl::PointXYZI& pt) -> Eigen::Vector3d { return c + ToVec3d(pt); }) /
                         source_->size();
        //std::cout << "source center: " << source_center_.transpose();
    }




    /// 基于gauss-newton的点面ICP
    bool AlignP2Plane(Sophus::SE3d& init_pose);
    /// 基于gauss-newton的点点ICP
    bool AlignP2P(Sophus::SE3d& init_pose);
    

   private:
    // 建立目标点云的Kdtree
    void BuildTargetKdTree();

    std::shared_ptr<KdTree> kdtree_ = std::make_shared<KdTree>();  // 第5章的kd树

    pcl::PointCloud<pcl::PointXYZI>::Ptr target_ = nullptr;
    pcl::PointCloud<pcl::PointXYZI>::Ptr source_ = nullptr;

    Eigen::Vector3d target_center_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d source_center_ = Eigen::Vector3d::Zero();

    bool gt_set_ = false;  // 真值是否设置
    Sophus::SE3d gt_pose_;

    Options options_;
};


#endif