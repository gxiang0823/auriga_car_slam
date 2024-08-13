#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h> // 用于ROS消息和PCL点云格式之间的转换
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h> 
#include <pcl/filters/project_inliers.h> 
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/LU>



// 定义栅格信息的数据结构
struct GridInfo {
    int point_count;  // 多少个点云落入
    float max_z;      // 最大z值
    float min_z;      // 最小z值
    int flag;         // 设定flag,0为未知，1为地面，2为路沿,3为不知道什么障碍物，初始化为0
    double height_differnece; //高度差，初始化为-1
    float x;   //存储珊格中点云的平均x值 ，全部初始化为0
    float y;   //存储珊格中点云的平均y值，全部初始化为0

    GridInfo() : point_count(0), max_z(std::numeric_limits<float>::min()), min_z(std::numeric_limits<float>::max()),
    flag(0),height_differnece(-1.0),x(0),y(0)
    {}
};

// 定义栅格地图的大小和分辨率
const float map_resolution = 0.1; // 每个栅格的分辨率
const int map_size_x = 200;       // 地图的宽度（以栅格为单位）
const int map_size_y = 200;       // 地图的高度（以栅格为单位）


// 初始化栅格信息地图
//这是一个map_size_x*map_size_y大小的地图，每个珊格是一个GridInfo数据类型
std::vector<std::vector<GridInfo>> grid_info_map(map_size_x, std::vector<GridInfo>(map_size_y)); 



// 定义栅格地图相关参数
double LidAngRlu = 0.2; // 0.2角度，不是弧度
    //转换成弧度
double angle_radians = LidAngRlu * M_PI /180.0;
double expansion_radius = 0.035; //按照0.2度乘以10算出来
bool occupancy_grid_map[map_size_x][map_size_y];
std::vector<double> ranges(2*M_PI/angle_radians); //储存不同角度轴下的距离



//定义发布者对象
ros::Publisher map_publisher;
ros::Publisher after_cloud_publisher;

//定义发布帧率
float Pub_Fre = 0.1;


//定义把车体和激光雷达包含的长方体的长宽高参数
float x = 4;
float y = 2;
float z = 8;
float RadarToFront = 1;


//定义直通滤波参数,以雷达坐标系作为原点，车辆前进方向为x轴，前进的左侧为y轴，往上是z轴
//这里是将保留的点云范围进行定义
float str_filt_plus_x = 10; 
float str_filt_minus_x = -10;
float str_filt_plus_y = 10;
float str_filt_minus_y = -10;
float str_filt_plus_z = 10;
float str_filt_minus_z = -10;


//定义半径滤波参数
float radius = 5;
float pointsInRadius = 5;



//定义去除地面点云时RANSAC方法参数
int max_iteration = 5000;
double threshold = 0.06;
double LidarFromTheGround = 0.78;
double ShoulderHeight = 0.15;
double DisFrCurb = 0;
double ShoulderHeightThresholds = 0.05;
double AllowableError = 0.05;

//地面点云初步去除
pcl::PointCloud<pcl::PointXYZ>::Ptr removePlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double LidarFrTheGnd, double ShdHet) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 筛选出z值在地面阈值的点云
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (cloud->points[i].z >= -(LidarFrTheGnd+ShdHet) && cloud->points[i].z <= -(LidarFrTheGnd-ShdHet)
            && cloud->points[i].y >= -DisFrCurb) {
            filteredCloud->push_back(cloud->points[i]);
        }
    }
    // 确保至少有3个点
    if (filteredCloud->size() < 3) {
        // 处理点云过少的情况，例如抛出异常或返回空指针
        ROS_ERROR("找不到三个点");
    }
    // 执行RANSAC算法，把筛选后点的点云输入seg做分割
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iteration);
    seg.setDistanceThreshold(threshold); 
    seg.setInputCloud(filteredCloud);
    // 调用seg.segment，传递初始化点
    seg.segment(*inliers, *coefficients);
    // 从原始点云中把所有内点分离出来
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    // 把索引更新
    pcl::PointIndices::Ptr new_inliers(new pcl::PointIndices);
    for (size_t i = 0; i < cloud->size(); ++i) {
    float distance = fabs(coefficients->values[0] * cloud->points[i].x +
                           coefficients->values[1] * cloud->points[i].y +
                           coefficients->values[2] * cloud->points[i].z +
                           coefficients->values[3]) /
                     sqrt(coefficients->values[0] * coefficients->values[0] +
                          coefficients->values[1] * coefficients->values[1] +
                          coefficients->values[2] * coefficients->values[2]);
    if (distance < threshold) {
        new_inliers->indices.push_back(i);
    }
}
    // 传入新索引和原始数据
    extract.setIndices(new_inliers);
    extract.setNegative(true); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr remainingCloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*remainingCloud);

    return remainingCloud;
}

//半径滤波
pcl::PointCloud<pcl::PointXYZ>::Ptr radiusFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (cloud->size() > 0)
    {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;  //创建半径滤波器
	radiusoutlier.setInputCloud(cloud); //输入点云
	radiusoutlier.setRadiusSearch(radius);   //搜索半径
	radiusoutlier.setMinNeighborsInRadius(pointsInRadius);//邻域点阈值
	radiusoutlier.filter(*cloud_filtered);
    return cloud_filtered;
    }
}

//直通滤波
pcl::PointCloud<pcl::PointXYZ>::Ptr passthroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (cloud->size() > 0)
    {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    // 创建滤波器对象
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);                                       // 设置待滤波的点云
    pass.setFilterFieldName("x");                                   // 设置在x轴方向上进行滤波
    pass.setFilterLimits(str_filt_minus_x, str_filt_plus_x);      // 设置滤波范围,在范围之外的点会被剪除
    pass.filter(*cloud_filtered);                                   // 开始过滤

    pass.setInputCloud(cloud_filtered);                                       // 设置待滤波的点云
    pass.setFilterFieldName("y");                                   // 设置在y轴方向上进行滤波
    pass.setFilterLimits(str_filt_minus_y, str_filt_plus_y);        // 设置滤波范围,在范围之外的点会被剪除
    pass.filter(*cloud_filtered);                                   // 开始过滤

    pass.setInputCloud(cloud_filtered);                                       // 设置待滤波的点云
    pass.setFilterFieldName("z");                                   // 设置在z轴方向上进行滤波
    pass.setFilterLimits(str_filt_minus_z, str_filt_plus_z);           // 设置滤波范围,在范围之外的点会被剪除
    // pass.setFilterLimitsNegative(true);                          // 是否反向过滤，默认为false
    pass.filter(*cloud_filtered);                                   // 开始过滤
    return cloud_filtered;
    }
}

//去除打到车体的点云
pcl::PointCloud<pcl::PointXYZ>::Ptr removeCarPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr,double x,double y,double z,double RTF) {
    if (point_cloud_ptr->size() > 0)
    {
    //创建空的PointIndices对象，用于存储需要删除的点的索引 
    pcl::PointCloud<pcl::PointXYZ>::iterator it;
    for (it = point_cloud_ptr->begin(); it != point_cloud_ptr->end();) {
        if ((*it).x <= RTF && (*it).x >= -x && (*it).y <= y/2 && (*it).y >= -y/2 && (*it).z <= z/2 && (*it).z >= -z/2 ) {
            it = point_cloud_ptr->erase(it);
        } 
        else {
            ++it;
        }
    }
    return point_cloud_ptr;
    }
}



//发布点云数据
void publishOccupancyGridMap() {
    nav_msgs::OccupancyGrid occupancy_grid_msg;
    occupancy_grid_msg.header.stamp = ros::Time::now();
    occupancy_grid_msg.header.frame_id = "velodyne";
    occupancy_grid_msg.info.map_load_time = ros::Time::now();
    occupancy_grid_msg.info.resolution = map_resolution;
    occupancy_grid_msg.info.width = map_size_x;
    occupancy_grid_msg.info.height = map_size_y;
    occupancy_grid_msg.info.origin.position.x = -0.5*map_size_x*map_resolution;
    occupancy_grid_msg.info.origin.position.y = -0.5*map_size_y*map_resolution;
    occupancy_grid_msg.info.origin.position.z = 0.0;
    occupancy_grid_msg.info.origin.orientation.x = 0.0;
    occupancy_grid_msg.info.origin.orientation.y = 0.0;
    occupancy_grid_msg.info.origin.orientation.z = 0.0;
    occupancy_grid_msg.info.origin.orientation.w = 1.0;
    for (int x = 0; x < map_size_x; ++x) {
        for (int y = 0; y < map_size_y; ++y) {
            if (occupancy_grid_map[x][y]) {
                occupancy_grid_msg.data.push_back(100); // occupied
            } else {
                occupancy_grid_msg.data.push_back(0); // free
            }
        }
    }
    map_publisher.publish(occupancy_grid_msg);
    
}


// 有需要可输出点云数据
void writePointCloudToFile(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& filename) {
    std::ofstream outFile(filename.c_str());
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing!" << std::endl;
        return;
    }

    for (const auto& point : *cloud) {
        outFile << point.x << " " << point.y << " " << point.z << std::endl;
    }

    outFile.close();
}

void convertPointCloudToOccupancyGridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
 {
    // 清空地图
    memset(occupancy_grid_map, false, sizeof(occupancy_grid_map));
    for(auto &range : ranges){
        range = 1000;
    }


    

    // 遍历点云中的每个点
    for (const auto& point : point_cloud->points) {
        // 计算点在地图中的栅格坐标(珊格坐标含义是第几行第几列)，按此法建立的珊格地图，激光雷达在珊格地图最中间
        // int grid_x = (int)(point.x / map_resolution + map_size_x / 2 -1);
        // int grid_y = (int)(point.y / map_resolution + map_size_y / 2  -1);
        int angle_axle = std::floor((atan2(point.x,point.y)+M_PI)/angle_radians);
        double range2 = std::pow(point.x,2)+std::pow(point.y,2);
        ranges[angle_axle] = std::min(ranges[angle_axle],range2);
    }

    //遍历栅格地图中的每一个格子
    for (int i = 0; i<map_size_x;i++){
        for(int j=0;j<map_size_y;j++){
            double grid_x = (j+0.5 - map_size_x / 2)*map_resolution;
            double grid_y = (i+0.5 - map_size_y / 2)*map_resolution;
            double grid_range2 = std::pow(grid_x,2)+std::pow(grid_y,2);
            int grid_angle_axle = std::floor((atan2(grid_x,grid_y)+M_PI)/angle_radians);
            int grid_angle_axle_p = grid_angle_axle+1;
            if (grid_angle_axle+1>=ranges.size()){
                int grid_angle_axle_p = 0;
            }
            if(grid_range2>ranges[grid_angle_axle]||grid_range2>ranges[grid_angle_axle_p]){
                occupancy_grid_map[i][j] = true;
            }
        }
    }

}


// 使用rviz对点云进行可视化
void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // 创建ROS消息对象
    sensor_msgs::PointCloud2 ros_cloud;

    // 将PCL点云转换为ROS消息
    pcl::toROSMsg(*cloud, ros_cloud);

    // 设置消息的元数据
    ros_cloud.header.frame_id = "base_link"; // 设置点云的坐标系，需要根据实际情况修改
    ros_cloud.header.stamp = ros::Time::now(); // 设置消息的时间戳

    // 发布点云消息
    after_cloud_publisher.publish(ros_cloud); // 发布消息
}

void doMsg(const sensor_msgs::PointCloud2::ConstPtr& msg_p){
    ROS_INFO("Here is the message");
    pcl::PointCloud<pcl::PointXYZ>::Ptr t_pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg_p,*t_pclCloud);
    //去除NAN的点
    std::vector<int> indices; //indices用来保存去除后的点的索引
    t_pclCloud->is_dense = false;  //若使用push_back创建点云，过滤nan值前需添加这个
    pcl::removeNaNFromPointCloud(*t_pclCloud, *t_pclCloud, indices);    
    // 去除打到车体的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr after_rev_car_filtered_cloud = removeCarPoints(t_pclCloud,x,y,z,RadarToFront);
    // 进行直通滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr after_strfil_filtered_cloud = passthroughFilter(after_rev_car_filtered_cloud);


    //进行地面点云分割
    pcl::PointCloud<pcl::PointXYZ>::Ptr after_rev_ground_cloud = removePlane(after_strfil_filtered_cloud,LidarFromTheGround,ShoulderHeight);
        // 进行半径滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr after_radius_filtered_cloud = radiusFilter(after_rev_ground_cloud);
    
    // 可视化点云
    //visualizePointCloud(after_radius_filtered_cloud);
    
    // 将点云数据转换为占用栅格地图
    convertPointCloudToOccupancyGridMap(after_radius_filtered_cloud);

    //发布地图
    publishOccupancyGridMap();




    // // 有需要可输出第一帧的点云数据
    // writePointCloudToFile(PC_at_SldHet,"raw_data.txt");
    // ros::shutdown();
}

int main(int argc, char  *argv[])
{
    setlocale(LC_ALL,"");
    //初始化 ROS 节点:命名(唯一)
    ros::init(argc,argv,"sub_pcloud");
    //实例化 ROS 句柄
    ros::NodeHandle nh;
    //实例化 订阅者 对象
    ros::Subscriber sub = nh.subscribe("/velodyne_points",1,doMsg);
    //发布珊格地图
    map_publisher = nh.advertise<nav_msgs::OccupancyGrid>("grid_map", 1);
    after_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("visulization_pointcloud", 1);

    //以一定帧率发布地图
    ros::Rate loop_rate(Pub_Fre);
    //设置循环调用回调函数
    ros::spin();//循环读取接收的数据，并调用回调函数处理
    loop_rate.sleep();
    return 0;
}
