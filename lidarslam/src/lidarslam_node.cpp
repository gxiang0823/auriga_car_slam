#include "ros/ros.h"
#include "common/point_cloud_utils.h"
#include "common/kdtree.h"
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <chrono>
#include <nav_msgs/Odometry.h>
#include <can_msgs/Frame.h>


// 动态坐标发布
#include <tf/transform_broadcaster.h>	// 动态坐标

ros::Publisher pub; 
ros::Publisher tf_pub;
nav_msgs::Odometry actual_pose;
double total_time(0);
int total_times(0);
pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_map(new pcl::PointCloud<pcl::PointXYZI>);
Icp3d lidar_ode;
Sophus::SE3d pose;
geometry_msgs::TransformStamped transform;
Sophus::SE3d init_pose;
int map_index = 0;
int map_counter = 0;

union VehicleArray{
	uint8_t array1[2];
	int16_t array2;
};


void velcallback(const can_msgs::Frame::ConstPtr& msg){

	if(msg -> id == 0x193){
		std::vector<uint8_t> array;
		for(auto &pt:msg->data){
			array.push_back(pt);
		}
		VehicleArray steer_angle, velocity;
		double steer,velocity_s;
		steer_angle.array1[0] =  array[2]; steer_angle.array1[1] =  array[3];
		steer = (steer_angle.array2 - 750.0) / 9.5209;  //degree
		velocity.array1[0] = array[0]; velocity.array1[1] = array[1];
		velocity_s = velocity.array2 / 32.9542; //  m/s
			
		ROS_INFO("steer_angle: %f",steer);
		// ROS_INFO("velocity: %f",velocity_s);	
	}	  
}


void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    auto t1 = std::chrono::high_resolution_clock::now();


    //sensor_msgs的PointCloud2转化为PCL的PointCloud2
    pcl::PCLPointCloud2::Ptr cloud = fromROSmsgs(cloud_msg);


    //体素滤波
    pcl::PCLPointCloud2::Ptr cloud_filtered = VoxelGrid(cloud);
    
    // 将PCL的PointCloud2转化为PCL的PointXYZI
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_filtered_1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromPCLPointCloud2(*cloud_filtered, *pcl_cloud_filtered_1);
    pcl::fromPCLPointCloud2(*cloud, *pcl_cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_filtered = removeCarPoints_FC(pcl_cloud_filtered_1);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_dense(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*pcl_cloud, *pcl_cloud_dense, mapping);

    if(map_counter==1){
        lidar_ode.SetSource(pcl_cloud_filtered);
        if(!lidar_ode.AlignP2P(init_pose)){
            std::cout<<"P2Plane failed"<<std::endl;
        }
        
        pose.so3() = pose.so3().rotZ(pose.angleZ() + init_pose.angleZ());
        pose.translation() = pose * init_pose.translation();
        pose.translation().z() = 0;
        //pose.so3() *= init_pose.so3();
        //std::cout<<init_pose.translation().z()<<std::endl;
    }

    

    if (map_index<5){map_index++;}else{map_index = 0,map_counter = 1;}
    // for (auto it : pcl_cloud_filtered->points){
    //     Eigen::Vector3d ps = pose * ToVec3d(it);
        
    //     if (map_counter!=1){
    //         pcl::PointXYZI pt;
    //         pt.x = ps[0];
    //         pt.y = ps[1];
    //         pt.z = ps[2];
    //         pt.intensity = it.intensity;
    //         pcl_cloud_map->points.push_back(pt);
    //     }else if(map_index==0&&map_counter==1){
    //         pcl::PointXYZI pt;
    //         pt.x = ps[0];
    //         pt.y = ps[1];
    //         pt.z = ps[2];
    //         pt.intensity = it.intensity;
    //         pcl_cloud_map->points.push_back(pt);
    //         pcl_cloud_map->points.erase(pcl_cloud_map->points.begin());
    //     }
    // }

    lidar_ode.SetTarget(pcl_cloud_filtered);
    
    	
    
    transform.header.stamp = cloud_msg->header.stamp;
    transform.header.frame_id = "map";
    transform.child_frame_id = "velodyne";
    transform.transform.translation.x = pose.translation().x();
    transform.transform.translation.y = pose.translation().y();
    transform.transform.translation.z = pose.translation().z();
    transform.transform.rotation.x = pose.unit_quaternion().x();
    transform.transform.rotation.y = pose.unit_quaternion().y();
    transform.transform.rotation.z = pose.unit_quaternion().z();
    transform.transform.rotation.w = pose.unit_quaternion().w();




    auto t2 = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    total_times++;
    // ROS_INFO("average time = %fms",total_time/total_times);
    //发布PCLpointcloud2格式的点云
    sensor_msgs::PointCloud2 output;
    pcl::PCLPointCloud2::Ptr cloud_map(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2 (*pcl_cloud_filtered, *cloud_map);
    pcl_conversions::fromPCL(*cloud_map, output);
    output.header.frame_id = "velodyne";
    // Publish the data
    //pub.publish (output);


    actual_pose.pose.pose.position.x = pose.translation().x();
	actual_pose.pose.pose.position.y = pose.translation().y();
	actual_pose.pose.pose.position.z = pose.translation().z();
	actual_pose.pose.pose.orientation.x = pose.unit_quaternion().x();
	actual_pose.pose.pose.orientation.y = pose.unit_quaternion().y();
    actual_pose.pose.pose.orientation.z = pose.unit_quaternion().z();
	actual_pose.pose.pose.orientation.w = pose.unit_quaternion().w();
	tf_pub.publish(actual_pose);//发布位姿
}

int main (int argc, char** argv)
{

    init_pose.transX(0);
    pose.transX(0);
    init_pose.so3() = Sophus::SO3d::exp(Eigen::Vector3d::Zero());
    // Initialize ROS
    ros::init (argc, argv, "my_pcl_tutorial");
    tf::TransformBroadcaster br;
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, cloud_cb);
    ros::Subscriber vel_sub = nh.subscribe<can_msgs::Frame>("/received_messages", 1, velcallback);


    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<sensor_msgs::PointCloud2> ("filtered_points", 1);
    tf_pub = nh.advertise<nav_msgs::Odometry> ("/ode_tf", 1);

    ros::Rate rate(10);  
    while (ros::ok()){
        // Spin
        ros::spinOnce();
        br.sendTransform(transform);
        rate.sleep();
    }
    
    return 0;
}
