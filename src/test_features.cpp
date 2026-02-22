#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/common/transforms.h>


#include <boost/program_options.hpp>
#include <Eigen/Eigen>
#include<boomer2_tools/Descriptor.hh>


namespace po = boost::program_options;
using namespace graph_matcher;

VectorFeaturePtr get_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string feature_type) {


    float normals_radius = 0.20;
    int pixel_n = 40;
    
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 
    ne.setRadiusSearch (normals_radius);

    // Compute the normals
    ne.compute (*cloud_normals);
    //search for neighborhood
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    
    // Calculate the center of the point cloud
    pcl::PointXYZ center;
    center.x = 0.0;
    center.y = 0.0;
    center.z = 0.0;
    int numPoints = cloud->size();
    for (const auto& point : *cloud) {
        center.x += point.x;
        center.y += point.y;
        center.z += point.z;
    }
    center.x /= numPoints;
    center.y /= numPoints;
    center.z /= numPoints;
    
    // find neighbors    
    // std::cerr << "find neighbors..." << std::endl;
    // std::cerr << pt << std::endl;
    std::vector<FeatureBasePtr> features;
    std::vector<cv::Point> keypoints_new;

    if ( tree->radiusSearch (center, normals_radius*2, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
    {
        // writer.write (fname, cloud_with_normals);
        // pcl::io::savePLYFileASCII(fname, cloud_with_normals);
        // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visualization"));
        // viewer->setBackgroundColor(0, 0, 0);
        // viewer->addPointCloud<pcl::PointXYZ>(cloud_with_normals, "cloud_with_normals");
        
        
        VectorFeaturePtr f1 (new VectorFeature());
        Eigen::Vector3d v1 = center.getVector3fMap().cast<double>();
        std::cerr << "For point " << v1.transpose() << " neighbors are " << pointIdxRadiusSearch.size() << std::endl;        f1->pos=v1; 
        f1->id = 1;

        if(feature_type == "PFH") {

            pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
            Eigen::VectorXf pfh_histogram(125);
            pfh.setRadiusSearch(0.15);
            pfh.computePointPFHSignature(*cloud,*cloud_normals, pointIdxRadiusSearch, 5, pfh_histogram);
            
            // std::cerr<<"For point "<<v1.transpose()<<" neighbors are "<<pointIdxRadiusSearch.size()<<std::endl;
            // std::cerr<<"Histogram "<<pfh_histogram.transpose()<<std::endl;
            
            //add feature to result                
            f1->feature = pfh_histogram.cast<double>();
            std::cout << "f1->feature: " << f1->feature.transpose() << std::endl;
        } else if (feature_type == "FPFH") {  
            // ...
            // fpfh.setRadiusSearch (0.15);

            // // Compute the features
            // fpfh.compute (*fpfhs);
        }

        if(f1->feature.norm() > 1e-2) 
        {   //ignore descriptors that are too tiny
            FeatureBasePtr f1_base(f1);
            features.push_back(f1_base);
            
            // keypoints_new.push_back(center);
        } 
        
    }
    else 
    {
    std::cerr << "No neighbors found for point " << center << std::endl;
    }

}

int main(int argc, char** argv) {

    std::string input_file, feature_type;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("input", po::value<std::string>(&input_file), "name of the input pcd file")
    ("feature_type", po::value<std::string>(&feature_type)->default_value("PFH"), "featue type: PFH, FPFH")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    


    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
        PCL_ERROR("Couldn't read file \n");
        return -1;
    }

    
    VectorFeaturePtr f1, f2;

    f1 = get_feature(cloud, feature_type);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*cloud, *cloud_rotated, transform);

    f2 = get_feature(cloud_rotated, feature_type);
    return 0;
}
