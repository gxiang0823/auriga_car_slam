#include "point_cloud_utils.h"



bool FitPlane(std::vector<Eigen::Matrix<float, 3, 1>>& data, Eigen::Matrix<float, 4, 1>& plane_coeffs, float eps) {
    if (data.size() < 3) {
        return false;
    }

    Eigen::MatrixXf A(data.size(),4);
    for (int i = 0; i < data.size(); ++i) {
        A.row(i).head<3>() = data[i].transpose();
        A.row(i)[3] = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
    plane_coeffs = svd.matrixV().col(3);

    // check error eps
    for (int i = 0; i < data.size(); ++i) {
        double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
        if (err * err > eps) {
            return false;
        }
    }
    return true;
}

bool FitPlane(std::vector<Eigen::Matrix<double, 3, 1>>& data, Eigen::Matrix<double, 4, 1>& plane_coeffs, float eps) {
    if (data.size() < 3) {
        return false;
    }

    Eigen::MatrixXd A(data.size(),4);
    for (int i = 0; i < data.size(); ++i) {
        A.row(i).head<3>() = data[i].transpose();
        A.row(i)[3] = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
    plane_coeffs = svd.matrixV().col(3);

    // check error eps
    for (int i = 0; i < data.size(); ++i) {
        double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
        if (err * err > eps) {
            return false;
        }
    }
    return true;
}

pcl::PCLPointCloud2::Ptr fromROSmsgs(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2); 
    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);
    return cloud;
}

pcl::PCLPointCloud2::Ptr VoxelGrid(const pcl::PCLPointCloud2::Ptr cloud){
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2); 
    sor.setInputCloud (cloudPtr);
    sor.setLeafSize (0.1, 0.1, 0.1);
    sor.filter (*cloud_filtered);
    // pcl::PassThrough<pcl::PCLPointCloud2> pass;
    // pass.setInputCloud(cloud_filtered);                                       // 设置待滤波的点云
    // pass.setFilterFieldName("x");                                   // 设置在x轴方向上进行滤波
    // pass.setFilterLimits(-10, 10);      // 设置滤波范围,在范围之外的点会被剪除
    // pass.filter(*cloud_filtered);                                   // 开始过滤

    // pass.setInputCloud(cloud_filtered);                                       // 设置待滤波的点云
    // pass.setFilterFieldName("y");                                   // 设置在y轴方向上进行滤波
    // pass.setFilterLimits(-5, 5);        // 设置滤波范围,在范围之外的点会被剪除
    // pass.filter(*cloud_filtered);                                   // 开始过滤

    // pass.setInputCloud(cloud_filtered);                                       // 设置待滤波的点云
    // pass.setFilterFieldName("z");                                   // 设置在z轴方向上进行滤波
    // pass.setFilterLimits(-20, 20);           // 设置滤波范围,在范围之外的点会被剪除
    // // pass.setFilterLimitsNegative(true);                          // 是否反向过滤，默认为false
    // pass.filter(*cloud_filtered);                                   // 开始过滤
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr removeCarPoints_FC(const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud_ptr,double x,double y,double z,double RTF) {
    //创建空的PointIndices对象，用于存储需要删除的点的索引 
    pcl::PointCloud<pcl::PointXYZI>::iterator it;
    for (it = point_cloud_ptr->begin(); it != point_cloud_ptr->end();) {
        if ((*it).x <= RTF && (*it).x >= -x && (*it).y <= y/2 && (*it).y >= -y/2 && (*it).z <= z/2 && (*it).z >= -z/2 ) {
            it = point_cloud_ptr->erase(it);
            continue;
        } 
        // else if ((*it).z>-0.7&&(*it).z<2&&(*it).x>1&&fabs((*it).y)<3){
        //     it = point_cloud_ptr->erase(it);
        //     continue;
        // }
        //else if ((*it).z<1.5&&fabs((*it).x)<10&&fabs((*it).y)<5){
        //     it = point_cloud_ptr->erase(it);
        //     continue;
        // } 
        else
         {
            ++it;
        }
    }
    return point_cloud_ptr;
}

bool Icp3d::AlignP2Plane(Sophus::SE3d &init_pose)
{
    //std::cout << "aligning with point to plane"<<std::endl;
    assert(target_ != nullptr && source_ != nullptr);
    // 整体流程与p2p一致，读者请关注变化部分

    Sophus::SE3d pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 1, 6>> jacobians(index.size());
    std::vector<double> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Eigen::Vector3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
            if (nn.size() > 3) {
                // convert to eigen
                std::vector<Eigen::Vector3d> nn_eigen;
                for (int i = 0; i < nn.size(); ++i) {
                    nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                }

                Eigen::Vector4d n;
                if (!FitPlane(nn_eigen, n)) {
                    // 失败的不要
                    effect_pts[idx] = false;
                    return;
                }

                double dis = n.head<3>().dot(qs) + n[3];
                if (fabs(dis) > options_.max_plane_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;
                //source_->points[idx].intensity = 1e10;

                // build residual
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * Sophus::SO3d::hat(q);
                J.block<1, 3>(0, 3) = n.head<3>().transpose();

                jacobians[idx] = J;
                errors[idx] = dis;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>& pre,
                                                                           int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    total_res += errors[idx] * errors[idx];
                    effective_num++;
                    return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            std::cout << "effective num too small: " << effective_num<<std::endl;
            return false;
        }

        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        pose.so3() = pose.so3() * Sophus::SO3d::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        //std::cout << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm()<<std::endl;

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            std::cout << "iter " << iter << " pose error: " << pose_error<<std::endl;
        }

        if (dx.norm() < options_.eps_) {
            //std::cout << "converged, dx = " << dx.transpose()<<std::endl;
            //std::cout<<pose.translation()<<std::endl;
            break;
        }
    }

    init_pose = pose;
    return true;
}

bool Icp3d::AlignP2P(Sophus::SE3d &init_pose)
{
    assert(target_ != nullptr && source_ != nullptr);

    Sophus::SE3d pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Eigen::Vector3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);  // 这里取1个最近邻
            if (!nn.empty()) {
                // convert to eigen
                Eigen::Vector3d p = ToVec3d(target_->points[nn[0]]);
                double dis2 = (p-qs).squaredNorm();

                if (dis2 > options_.max_nn_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;
                //source_->points[idx].intensity = 1e10;

                // build residual
                Eigen::Vector3d e = p-qs;
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = pose.so3().matrix() * Sophus::SO3d::hat(q);
                J.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

                jacobians[idx] = J;
                errors[idx] = e;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>& pre,
                                                                           int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    total_res += errors[idx].dot(errors[idx]);
                    effective_num++;
                    return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            std::cout << "effective num too small: " << effective_num<<std::endl;
            return false;
        }

        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        pose.so3() = pose.so3() * Sophus::SO3d::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        //std::cout << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm()<<std::endl;

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            std::cout << "iter " << iter << " pose error: " << pose_error<<std::endl;
        }

        if (dx.norm() < options_.eps_) {
            //std::cout << "converged, dx = " << dx.transpose()<<std::endl;
            //std::cout<<pose.translation()<<std::endl;
            break;
        }
    }

    init_pose = pose;
    return true;
}

void Icp3d::BuildTargetKdTree()
{   
    kdtree_->BuildTree(target_);
}
