/**
  * Simple utility to transform a scan from global to local coordinates, 
  * given a pcd file and scanner pose
  */
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Geometry>

#include <boost/program_options.hpp> 
namespace po = boost::program_options;

// Function to map intensity to an RGB value
void intensityToRGB(float intensity, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Normalize intensity between 0 and 255
    float normalized = intensity * 255.0f; // Assuming intensity is between 0 and 1

    // Convert normalized intensity to RGB (grayscale in this example)
    r = static_cast<uint8_t>(normalized);
    g = static_cast<uint8_t>(normalized);
    b = static_cast<uint8_t>(normalized);
}
// Convert XYZI to XYZRGB
pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertXYZIToXYZRGB(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_xyzi) {
    // Use boost::make_shared or new operator to allocate the point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>());

    cloud_xyzrgb->width = cloud_xyzi->width;
    cloud_xyzrgb->height = cloud_xyzi->height;
    cloud_xyzrgb->is_dense = cloud_xyzi->is_dense;
    cloud_xyzrgb->points.resize(cloud_xyzi->points.size());

    for (std::size_t i = 0; i < cloud_xyzi->points.size(); ++i) {
        const auto& point_i = cloud_xyzi->points[i];
        auto& point_rgb = cloud_xyzrgb->points[i];

        // Copy XYZ coordinates
        point_rgb.x = point_i.x;
        point_rgb.y = point_i.y;
        point_rgb.z = point_i.z;

        // Convert intensity to RGB
        // std::cerr<<point_i.x << ", " << point_i.y << ", " << point_i.z << ", "<<(point_i.intensity)<<std::endl;
        intensityToRGB(point_i.intensity, point_rgb.r, point_rgb.g, point_rgb.b);
    }

    return cloud_xyzrgb;
}

void downsample_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud, double grid) {
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    std::cout << "downsample cloud of " << cloud->size() << " points" << std::endl;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(grid, grid, grid);
    voxel_grid.filter(*downsampled_cloud);
}

int main(int argc, char** argv) {

  std::string in_file, out_file, pt_type, pos_file;
  bool load_pos = false;
  bool isLeft2Right = false;
  double scanner_x, scanner_y, scanner_z, scanner_roll, scanner_pitch, scanner_yaw;
  double axis_x, axis_y, axis_z, axis_angle;
  double grid = 0;
  std::string scan_id;
  po::options_description desc("Allowed options");
  desc.add_options() 	
    ("help", "produce help message") 	
    ("input", po::value<std::string>(&in_file), "name of the input pcd file") 	
    ("output", po::value<std::string>(&out_file), "name of the output pcd file") 	
    ("pt_type", po::value<std::string>(&pt_type)->default_value("XYZRGB"), "point cloud type, XYZRGB or XYZI, (or XYZ)")
    ("scanner_x", po::value<double>(&scanner_x)->default_value(0.0), "X position of scanner")
    ("scanner_y", po::value<double>(&scanner_y)->default_value(0.0), "Y position of scanner")
    ("scanner_z", po::value<double>(&scanner_z)->default_value(0.0), "Z position of scanner")
    ("scanner_roll", po::value<double>(&scanner_roll)->default_value(0.0), "scanner roll in DEGREE")
    ("scanner_pitch", po::value<double>(&scanner_pitch)->default_value(0.0), "scanner pitch in DEGREE")
    ("scanner_yaw", po::value<double>(&scanner_yaw)->default_value(0.0), "scanner yaw in DEGREE")
    ("pos_file", po::value<std::string>(&pos_file), "name of the scan pos file to read") 	
    ("grid", po::value<double>(&grid), "voxel grid size")
    ("to_global", "from local to global")
    ("left2right", "left coordinate to right coordinate")
  ;

  po::variables_map vm;     
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  bool is2Global = false;

  if (vm.count("left2right"))
  {
    std::cout <<"convert the point cloud from left to right\n";
    isLeft2Right = true;
  }
  if (vm.count("help"))     
  { 	
    std::cout << desc << "\n"; 	
    return 1;     
  }

  if (vm.count("to_global"))
  {
    std::cout <<"convert the point cloud to global frame";
    is2Global = true;
  }

  if (!vm.count("input") || !vm.count("output"))     
  {  
    std::cout << "Please provide an input and output file name tags. Run with --help for usage.\n";	
    return -1;     
  }
  if (vm.count("pos_file"))     
  {  
    std::cout << "scan pos file is given. " << pos_file << "\n";	
    load_pos = true;
  }

  Eigen::Affine3d T;
  if (load_pos) {
    std::ifstream pos_f(pos_file);
    if(!pos_f.is_open()) {
      std::cerr<<"Error opening pos file\n";
      return -1;
    }
    std::string line;
    if(std::getline(pos_f, line)) {
      std::cerr<<"Read line: "<<line<<std::endl;
      std::istringstream iss(line);
      iss >> scan_id >> scanner_x >> scanner_y >> scanner_z
            >> axis_x >> axis_y >> axis_z >> axis_angle;
    } else {
      std::cerr<<"Error reading pos file\n";
      return -1;
    }
    
    
    pos_f.close();

    std::cerr<<"Scan ID: "<<scan_id<<std::endl;
    std::cerr<<"Scanner pose: "<<scanner_x<<", "<<scanner_y<<", "<<scanner_z<<std::endl;
    std::cerr<<"Axis: "<<axis_x<<", "<<axis_y<<", "<<axis_z<< ", " << axis_angle<< std::endl;

    Eigen::Vector3d axis(axis_x, axis_y, axis_z);
    T =
    Eigen::Translation<double, 3>(scanner_x, scanner_y, scanner_z) *
    Eigen::AngleAxis<double>(axis_angle, axis);
  }
  else {
    T =
    Eigen::Translation<double, 3>(scanner_x, scanner_y, scanner_z) *
    Eigen::AngleAxis<double>(M_PI*scanner_roll/180., Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxis<double>(M_PI*scanner_pitch/180., Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxis<double>(M_PI*scanner_yaw/180., Eigen::Vector3d::UnitZ());
  }

  std::cerr << "Scanner pose T=\n"
            << T.matrix() << "\n";

  Eigen::Affine3d Tinv = T.inverse();
  std::cerr << "inverse T=\n"
            << Tinv.matrix() << "\n";

  //pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
  pcl::PCDReader reader;

  if(pt_type == "XYZI") {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
    if(reader.read(in_file, *cloud_xyzi) < 0) {
      std::cerr<<"Error reading pcd file\n";
      return -1;
    }

    // Convert to XYZRGB
    cloud = convertXYZIToXYZRGB(cloud_xyzi);

  }
  else {
    if(reader.read(in_file, *cloud) < 0) {
      std::cerr<<"Error reading pcd file\n";
      return -1;
    }
  }




  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for(auto it=cloud->begin(); it!=cloud->end(); it++) {
    centroid+= Eigen::Vector3d(it->x, it->y, it->z) / cloud->size();
  }
  std::cerr<<"Centroid before transform: "<<centroid.transpose()<<std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

  if (is2Global) {
pcl::transformPointCloud (*cloud, *transformed_cloud, T);
  }
  else {
    pcl::transformPointCloud (*cloud, *transformed_cloud, Tinv);
  }
  

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB> ());
  // auto T_trans = T;
  // T_trans.linear() = Eigen::Matrix3d::Identity();
  // std::cerr << T_trans.matrix() << std::endl;
  // auto T_rot = T;
  // T_rot.translation() = Eigen::Vector3d::Zero();
  // std::cerr << T_rot.matrix() << std::endl;

  // pcl::transformPointCloud (*cloud, *transformed_cloud_2, T_trans.inverse());
  // pcl::transformPointCloud (*transformed_cloud_2, *transformed_cloud_2, T_rot.inverse());

  

  

  if (isLeft2Right) {
    for(auto it=transformed_cloud->begin(); it!=transformed_cloud->end(); it++) {
      it->x = -it->x;
      it->y = -it->y;
    }
  }

  Eigen::Vector3d centroid_t = Eigen::Vector3d::Zero();
  for(auto it=transformed_cloud->begin(); it!=transformed_cloud->end(); it++) {
    centroid_t+= Eigen::Vector3d(it->x, it->y, it->z) / transformed_cloud->size();
  }
  std::cerr<<"Centroid after transform: "<<centroid_t.transpose()<<std::endl;

  pcl::PCDWriter writer;
  if (grid>0){
    // Split the point cloud into 4 parts
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_3(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_4(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_5(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_6(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_7(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_8(new pcl::PointCloud<pcl::PointXYZRGB>());

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();

    int discard_count = 0;
    for (const auto& point : transformed_cloud->points) {
        float dx = point.x - centroid_t.x();
        float dy = point.y - centroid_t.y();
        float dz = point.z - centroid_t.z();
        float dist_2 = dx*dx + dy*dy + dz*dz;        
        if(dist_2 > 256)
        { 
          // distance are too far (16m away), ignore the points
          discard_count++;
          continue;
        }
        if (point.z >centroid_t.z()) 
        {
            if (point.x > centroid_t.x() && point.y > centroid_t.y()) {
                cloud_1->points.push_back(point);
            } else if (point.x > centroid_t.x() && point.y <= centroid_t.y()) {
                cloud_2->points.push_back(point);
            } else if (point.x <= centroid_t.x() && point.y > centroid_t.y()) {
                cloud_3->points.push_back(point);
            } else {
                cloud_4->points.push_back(point);
            }
        } 
        else 
        {
            if (point.x > centroid_t.x() && point.y > centroid_t.y()) {
                cloud_5->points.push_back(point);
            } else if (point.x > centroid_t.x() && point.y <= centroid_t.y()) {
                cloud_6->points.push_back(point);
            } else if (point.x <= centroid_t.x() && point.y > centroid_t.y()) {
                cloud_7->points.push_back(point);
            } else {
                cloud_8->points.push_back(point);
            }
        }
    }
    std::cout << "discard points: " << discard_count << std::endl;
    cloud_1->width = cloud_1->points.size();
    cloud_1->height = 1;
    cloud_1->is_dense = true;

    cloud_2->width = cloud_2->points.size();
    cloud_2->height = 1;
    cloud_2->is_dense = true;

    cloud_3->width = cloud_3->points.size();
    cloud_3->height = 1;
    cloud_3->is_dense = true;

    cloud_4->width = cloud_4->points.size();
    cloud_4->height = 1;
    cloud_4->is_dense = true;

    cloud_5->width = cloud_5->points.size();
    cloud_5->height = 1;
    cloud_5->is_dense = true;

    cloud_6->width = cloud_6->points.size();
    cloud_6->height = 1;
    cloud_6->is_dense = true;

    cloud_7->width = cloud_7->points.size();
    cloud_7->height = 1;
    cloud_7->is_dense = true;

    cloud_8->width = cloud_8->points.size();
    cloud_8->height = 1;
    cloud_8->is_dense = true;



    // Optionally, save the split clouds to separate files
    downsample_cloud(cloud_1, cloud_1, grid);
    downsample_cloud(cloud_2, cloud_2, grid);
    downsample_cloud(cloud_3, cloud_3, grid);
    downsample_cloud(cloud_4, cloud_4, grid);
    downsample_cloud(cloud_5, cloud_5, grid);
    downsample_cloud(cloud_6, cloud_6, grid);
    downsample_cloud(cloud_7, cloud_7, grid);
    downsample_cloud(cloud_8, cloud_8, grid);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB> ());
    *downsampled_cloud = *cloud_1 + *cloud_2 + *cloud_3 + *cloud_4 + *cloud_5 + *cloud_6 + *cloud_7 + *cloud_8;
    writer.write (out_file, *downsampled_cloud);
  }
  else {
    writer.write (out_file, *transformed_cloud);
  }
  

  
  // centroid_t = Eigen::Vector3d::Zero();
  // for(auto it=transformed_cloud_2->begin(); it!=transformed_cloud_2->end(); it++) {
  //   centroid_t+= Eigen::Vector3d(it->x, it->y, it->z) / transformed_cloud_2->size();
  // }
  // std::cerr<<"Centroid after transform 2: "<<centroid_t.transpose()<<std::endl;
  // std::string to_replace = ".pcd";
  // std::string replace_with = "2.pcd";

  // size_t pos = out_file.find(to_replace);
  // if (pos != std::string::npos) {
  //     out_file.replace(pos, to_replace.length(), replace_with);
  // }
  // std::cerr   << "Writing to " << out_file << std::endl;
  // writer.write (out_file, *transformed_cloud_2);


  return 0;
}
