#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/pcl_config.h>

#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Geometry>

#include <sstream> // std::stringstream

#include <boomer2_tools/CloudRenderer.hh>
#include <boomer2_tools/GraphBuilder.hh>
#include <boomer2_tools/Detector.hh>
#include <boomer2_tools/Descriptor.hh>
#include <boomer2_tools/Logger.hh>


#include "omp.h"

#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>




namespace po = boost::program_options;

  using PointT = pcl::PointXYZRGB;
  using PointNormalT = pcl::PointXYZRGBNormal;
  
void drawKeypoints(cv::Mat &image, std::vector<cv::Point> &keypoints)
{

  // project and draw label points
  for (int p = 0; p < keypoints.size(); p++)
  {

    cv::Point pt = keypoints[p];

    // check if in FOV
    if (pt.x < 0 || pt.x >= image.rows ||
        pt.y < 0 || pt.y >= image.cols)
    {
      continue;
    }

    cv::Point ul, lr;
    ul.y = MAX(pt.x - 5, 0);
    ul.x = MAX(pt.y - 5, 0);
    lr.y = MIN(pt.x + 5, image.rows - 1);
    lr.x = MIN(pt.y + 5, image.cols - 1);
    cv::rectangle(image, ul, lr,
                  cv::Scalar(0, 255, 0), 2, cv::LINE_8);
  }
}

// Function to compute the grayscale value of a point based on its RGB values
uint8_t computeGrayscale(const pcl::PointXYZRGB& point) {
    return static_cast<uint8_t>(0.299 * point.r + 0.587 * point.g + 0.114 * point.b);
}

// Function to color a point cloud with a specific RGB tint while maintaining grayscale
void colorPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, uint8_t tint_r, uint8_t tint_g, uint8_t tint_b) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < cloud->points.size(); ++i) {
        // Calculate the original grayscale value
        uint8_t gray = computeGrayscale(cloud->points[i]);

        // Blend the grayscale with the tint color
        cloud->points[i].r = (gray * tint_r) / 255;
        cloud->points[i].g = (gray * tint_g) / 255;
        cloud->points[i].b = (gray * tint_b) / 255;
    }
}


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


graph_matcher::DescriptorExtractPtr getExtractor(std::shared_ptr<graph_matcher::CloudRenderer> & rend,
                                                  cv::Mat &intensity, 
                                                  cv::Mat &depth, 
                                                  std::string feature_type,
                                                  bool save_points=false,
                                                  std::string save_points_folder="")
{
  graph_matcher::DescriptorExtractPtr extractor;
  
  if (feature_type == "NONE" || feature_type == "CONNECT")
  {
    extractor = graph_matcher::DescriptorExtractPtr(new graph_matcher::EmptyExtract(depth, rend));
  }
  if (feature_type == "SIFT" ||
      feature_type == "BRISK" ||
      feature_type == "ORB")
  {
    cv::Mat intensity_grayscale;
    intensity.convertTo(intensity_grayscale, CV_8U, 255);
    extractor = graph_matcher::DescriptorExtractPtr(new graph_matcher::OpenCVExtract(
        depth, intensity_grayscale, feature_type, rend));
  }
  if (feature_type == "PFH" || feature_type == "FPFH" || feature_type == "SHOT" || feature_type == "NDT")
  {
    extractor = graph_matcher::DescriptorExtractPtr(new graph_matcher::PCLExtract(
        depth, feature_type, rend, save_points, save_points_folder));
  }
  // if (feature_type == "NN") {
  //   extractor = graph_matcher::DescriptorExtractPtr(new graph_matcher::NNExtract(
  //       depth, feature_type, rend, learned_features));
  // }

  return extractor;
}


void matchGraphs(graph_matcher::GraphBuilder &gb,
                             graph_matcher::FeatureGraphPtr &graph,
                             graph_matcher::FeatureGraphPtr &graph_second,
                             graph_matcher::RANSAC_options &options,
                             std::string feature_type,
                             bool do_connectivity,
                             std::vector<std::pair<int, int>> &matches,
                             Eigen::Affine3d &reg
                             )
{    
    if (feature_type == "CONNECT")
    {
      graph->convertToConnectivityFeatures();
      graph_second->convertToConnectivityFeatures();
    }

    if (do_connectivity)
    {
      graph->convertToConnectivityFeatures(true);
      graph_second->convertToConnectivityFeatures(true);
      std::cerr << "Using both feature of type " << feature_type << " and connectivity information\n";
    }    
    
    matches = gb.matchGraphs(graph, graph_second, options, reg);      
    
}
 // Function to write test results to JSON file
void writeTestResultToJson(double err_t, double err_r, double err_dt, double err_dr, const std::string& json_filename, const std::string& reg_type) {
  std::ostringstream json_stream;
  json_stream << "{\n";
  json_stream << "  \"method\": \"" << reg_type << "\",\n";
  json_stream << "    \"translation_error\": " << err_t << ",\n";
  json_stream << "    \"rotation_error_deg\": " << err_r << ",\n";
  json_stream << "    \"translation_drift_per_meter\": " << err_dt << ",\n";
  json_stream << "    \"rotation_drift_per_meter\": " << err_dr << "\n";
  json_stream << "}";

  std::ofstream json_file(json_filename);
  if (json_file.is_open()) {
    json_file << json_stream.str();
    json_file.close();
  }
}

 // Function to write test results time cost to JSON file
void writeTestTimeCostToJson(double time_cost_bolt_detection, double time_cost_ransac, double time_cost_icp, double total_time_cost, const std::string& json_filename, const std::string& reg_type) {
  std::ostringstream json_stream;
  json_stream << "{\n";
  json_stream << "  \"method\": \"" << reg_type << "\",\n";
  json_stream << "    \"time_cost_bolt_detection\": " << time_cost_bolt_detection << ",\n";
  json_stream << "    \"time_cost_ransac\": " << time_cost_ransac << ",\n";
  json_stream << "    \"time_cost_icp\": " << time_cost_icp << ",\n";
  json_stream << "    \"total_time_cost\": " << total_time_cost << "\n";
  json_stream << "}";

  std::ofstream json_file(json_filename);
  if (json_file.is_open()) {
    json_file << json_stream.str();
    json_file.close();
  }
}

int main(int argc, char **argv)
{ 

  std::string in_file, in_file_2, label_file, label_file_2, model_file, base_name, feature_type, pt_type, pos_file_1, pos_file_2, reg_type, feature_file, log_file;
  double cam_x, cam_y, cam_z,coffset_x,coffset_y,coffset_z,coffset_yaw; 
  double cam_x_gt, cam_y_gt, cam_z_gt; //fix me: loading two point clouds with known ground truth transformation, rotation is not handled.
  int image_x, image_y;

  int NX=3, NY=1, NYAW=1; 
  double x_bound = 3;
  double y_bound = 3;
  double yaw_bound = 2*M_PI;

  double masking_distance = -1.0;

  std::cout << "PCL Version: " << PCL_VERSION << std::endl;
  //------------RANSAC options--------------/
  graph_matcher::RANSAC_options options;
  options.match_tolerance = 0.1;
  options.max_iterations = 1000000;
  options.seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
  options.random_seed = true;
  options.refine_ICP = false;
  options.forward_reverse = false;
  options.do_simple_match = false;
  options.match_n_closest = false;
  options.dist_threshold = 0.1;        // if doing simple match, consider matches with a score below this
  options.second_closest_thresh = 0.9; // >=1 means just match to closest
  options.n_closest = 3;               // if doing closest n, these many n
  options.use_RANSAC = true;


  po::options_description desc("Allowed options");
  desc.add_options()
  ("help", "produce help message")
  ("batch-test", "run a benchmark test for the selected parameters over many offsets")
  ("batch-NX", po::value<int>(&NX)->default_value(3), "Number of X offsets to test")
  ("batch-NY", po::value<int>(&NY)->default_value(1), "Number of Y offsets to test")
  ("batch-NYAW", po::value<int>(&NYAW)->default_value(1), "Number of YAW offsets to test")
  ("batch-x-bound", po::value<double>(&x_bound)->default_value(3), "X offset bound")
  ("batch-y-bound", po::value<double>(&y_bound)->default_value(3), "Y offset bound")
  ("batch-yaw-bound", po::value<double>(&yaw_bound)->default_value(2*M_PI), "YAW offset bound")
  ("input", po::value<std::string>(&in_file), "name of the input pcd file")
  ("input_2", po::value<std::string>(&in_file_2), "name of the input pcd file")
  ("fisheye", "render a fisheye image") 	
  ("pt_type", po::value<std::string>(&pt_type)->default_value("XYZRGB"), "point cloud type, XYZ, XYZRGB or XYZI")
  ("depth-only", "detect bolts use depth image only") 	
  ("labels", po::value<std::string>(&label_file), "name of the pcd file that holds labeled points")
  ("labels_2", po::value<std::string>(&label_file_2), "name of the pcd file that holds labeled points")
  ("model", po::value<std::string>(&model_file), "file path of the model")
  ("cpu", "force to use cpu version")
  ("icp_refine", "force to use cpu version")
  ("feature", po::value<std::string>(&feature_type), "what kind of features to use in the graph. Options are: NONE, CONNECT, SIFT, BRISK, ORB, PFH, SHOT, NDT")
  ("feature_connect", "add graph connectivity as a dimension of the above feature")
  ("cam_x", po::value<double>(&cam_x)->default_value(0.0), "X initial position of camera")
  ("cam_y", po::value<double>(&cam_y)->default_value(0.0), "Y initial position of camera")
  ("cam_z", po::value<double>(&cam_z)->default_value(0.0), "Z initial position of camera")
  ("cam_x_gt", po::value<double>(&cam_x_gt)->default_value(0.0), "X initial position of camera")
  ("cam_y_gt", po::value<double>(&cam_y_gt)->default_value(0.0), "Y initial position of camera")
  ("cam_z_gt", po::value<double>(&cam_z_gt)->default_value(0.0), "Z initial position of camera")
  ("image_x", po::value<int>(&image_x)->default_value(520), "Image height to render")
  ("image_y", po::value<int>(&image_y)->default_value(520), "Image width to render")
  ("ransac_simple_match", "RANSAC with a simple threshold on feature distance")
  ("ransac_dist_thresh", po::value<double>(&options.dist_threshold)->default_value(0.1), "Threshold for associating features with simple distance metric in ransac")
  ("ransac_second_closest", "RANSAC with a test on the ratiobetween closest and second closest feature")
  ("ransac_second_closest_thresh", po::value<double>(&options.second_closest_thresh)->default_value(0.8), "Threshold for associating features with second closest ration metric in ransac")
  ("ransac_n_closest", "RANSAC with associating to the n closest features")
  ("ransac_n_closest_thresh", po::value<int>(&options.n_closest)->default_value(5), "Just associate to the N closest features in ransac")
  ("ransac_forward_reverse", "RANSAC keep only mtches both in forward and reverse direction")
  ("coffset_x", po::value<double>(&coffset_x)->default_value(3.0), "X initial position of camera")
  ("coffset_y", po::value<double>(&coffset_y)->default_value(-3.0), "Y initial position of camera")
  ("coffset_z", po::value<double>(&coffset_z)->default_value(0), "Z initial position of camera")
  ("coffset_yaw", po::value<double>(&coffset_yaw)->default_value(0), "Z initial position of camera")
  ("pos_file_1", po::value<std::string>(&pos_file_1), "name of the scan_1 pos file to read") 	
  ("pos_file_2", po::value<std::string>(&pos_file_2), "name of the scan_2 pos file to read") 	
  ("reg_type", po::value<std::string>(&reg_type)->default_value("RANSAC"), "registration type; RANSAC")
  ("load_features", po::value<std::string>(&feature_file), "load features from file")
  ("log_file_name", po::value<std::string>(&log_file), "log file name")
  ("save_graph", "save the graph to a file")
  ("connect_pts", "connect two point clouds")
  ("fix_seed", po::value<int>(&options.seed)->default_value(std::time(0)), "fix the random seed")
  ("show_img", "show the image with keypoints")
  ("masking_distance", po::value<double>(&masking_distance)->default_value(-1.0), "distance threshold for masking points based on forward direction")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  bool batch_test = vm.count("batch-test");
  bool load2ndPoints = vm.count("input_2");
  bool connect_pts = false;
  bool isLoadingFeatures = vm.count("load_features");
  bool save_graph = vm.count("save_graph");
  bool show_img = vm.count("show_img");
  std::string input_1_name = "";
  std::string input_2_name = "";
  size_t pos_1 = in_file.find_last_of("/\\");
  if (pos_1 != std::string::npos) {
    input_1_name = in_file.substr(pos_1 + 1, in_file.length() - pos_1 - 9);
  }
  size_t pos_2 = in_file_2.find_last_of("/\\");
  if (pos_2 != std::string::npos) {
    input_2_name = in_file_2.substr(pos_2 + 1, in_file_2.length() - pos_2 - 9);
  }

  if (save_graph) {
    std::cout << "Saving the graph to a file\n";    
  }


  if(load2ndPoints && vm.count("connect_pts")) {
    connect_pts = true;
  }
  // fix me: cam_x_gt is the translation between two point clouds, might need a better name
  if(!load2ndPoints) {
    if (cam_x_gt != 0 || cam_y_gt != 0 || cam_z_gt != 0) {
      cam_x_gt = 0;
      cam_y_gt = 0;
      cam_z_gt = 0;
      std::cout << "Loading only one point cloud, then cam_x_gt, cam_y_gt, cam_z_gt (translation between two point clouds) are not applicable, they are set to 0!\n";
    }    
  }
  bool customized_offset = vm.count("coffset_x") || vm.count("coffset_y") || vm.count("coffset_z") || vm.count("coffset_yaw");
  if (customized_offset)
  {
    std::cout << "Using customized offset:" << coffset_x << ", "<< coffset_y << ", "<< coffset_z << ", " << coffset_yaw <<std::endl;  
  }
  options.match_n_closest = vm.count("ransac_n_closest");
  options.do_simple_match = vm.count("ransac_simple_match") && (!options.match_n_closest);

  if (vm.count("ransac_second_closest"))
  {
    options.do_simple_match = false;
    options.match_n_closest = false;
  }

  if (vm.count("icp_refine"))
  {
    options.refine_ICP = true;
    std::cerr << "ICP refinement is enabled\n";
  }
  // nothing was selected, default to simple match
  if (vm.count("ransac_n_closest") + vm.count("ransac_second_closest") + vm.count("ransac_simple_match") == 0)
  {
    options.do_simple_match = true;
    options.match_n_closest = false;
  }

  options.forward_reverse = vm.count("ransac_forward_reverse");

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }
  if (!vm.count("input"))
  {
    std::cout << "Please provide an input and label file name tags. Run with --help for usage.\n";
    return -1;
  }

  if (!vm.count("model") && !vm.count("labels"))
  {
    std::cout << "Please provide a label file or a model file. Run with --help for usage.\n";
    return -1;
  }

  bool load_pos_1 = false;
  bool load_pos_2 = false;
  if (vm.count("pos_file_1"))     
  {  
    std::cout << "scan pos file is given. " << pos_file_1 << "\n";	
    load_pos_1 = true;
  }

  if (vm.count("pos_file_2"))     
  {  
    std::cout << "scan pos file is given. " << pos_file_2 << "\n";	
    load_pos_2 = true;
  }

  if (load_pos_1 != load_pos_2) {
    std::cout << "Please provide both pos files or none. Run with --help for usage.\n";
    return -1;
  }
  
  bool useModel = false;
  bool useCPU = false;
  bool depthOnly = false;

  
  if (vm.count("depth-only"))
  {
    depthOnly = true;
  }

  if (vm.count("model"))
  {
    useModel = true;
  }

  if (vm.count("cpu")) {
    useCPU = true;
  }

  bool useLabel = false;
  bool useLabel_2 = false;

  if (vm.count("labels"))
  {
    useLabel = true;
  }

  if (vm.count("labels_2"))
  {
    useLabel_2 = true;
  }

  if (reg_type != "RANSAC")
  {
    std::cerr << "Unkonw registration type specified. Choose one of the implemented\n";
    return -1;
  }

  if (reg_type != "RANSAC")
  {
    options.use_RANSAC = false;
  }
  if (feature_type != "NONE" &&
      feature_type != "CONNECT" &&
      feature_type != "SIFT" &&
      feature_type != "BRISK" &&
      feature_type != "ORB" &&
      feature_type != "PFH" &&
      feature_type != "FPFH" &&
      feature_type != "NDT" &&
      feature_type != "SHOT" &&
      feature_type != "NN" 
      )
  {
    std::cerr << "Unkonw feature type specified. Choose one of the implemented\n";
    return -1;
  }
  bool do_connectivity = (vm.count("feature_connect") > 0) && feature_type != "CONNECT";
  bool fisheye = vm.count("fisheye");

  // fix me: decide type automatically or better way
  // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
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
  else
  {
    if (reader.read(in_file, *cloud) < 0)
    {
      std::cerr << "Error reading pcd file\n";
      return -1;
    }
  }


  std::cerr << "PointCloud read: " << cloud->width * cloud->height
            << " data points \n";

  for (auto &pt : *cloud)
  {
    pcl::PointXYZ pt_xyz;
    pt_xyz.x = pt.x;
    pt_xyz.y = pt.y;
    pt_xyz.z = pt.z;

    cloud_xyz->push_back(pt_xyz);
  }

  // fix me: decide type automatically or better way
  // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_2(new pcl::PointCloud<pcl::PointXYZ>());
  if (load2ndPoints)
  {

    if(pt_type == "XYZI") {

      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
      if(reader.read(in_file_2, *cloud_xyzi) < 0) {
        std::cerr<<"Error reading pcd file\n";
        return -1;  
    }

      // Convert to XYZRGB
      cloud_2 = convertXYZIToXYZRGB(cloud_xyzi);
    }
    else
    {
      if (reader.read(in_file_2, *cloud_2) < 0)
      {
        std::cerr << "Error reading pcd file\n";
        return -1;
      }
    }

    for (auto &pt : *cloud_2)
    {
      pcl::PointXYZ pt_xyz;
      pt_xyz.x = pt.x;
      pt_xyz.y = pt.y;
      pt_xyz.z = pt.z;

      cloud_xyz_2->push_back(pt_xyz);
    }
  }


  graph_matcher::CamParams cp;
  cp.center_x = image_x/2.0;
  cp.center_y = image_y/2.0;
  cp.f = 250.0;
  cp.image_x = image_x;
  cp.image_y = image_y;
  cp.max_dist = 15;
  cp.isFisheye = fisheye;

  if(fisheye) cp.f = 350.0;

  // cv::namedWindow ("Intensity", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Graphs", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);
  // cv::namedWindow ("Depth", cv::WINDOW_AUTOSIZE);
  // cv::namedWindow ("Depth Second", cv::WINDOW_AUTOSIZE);

  cv::Mat depth = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(1.0));
  cv::Mat intensity = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(0.0));
  
  Eigen::Affine3d T;
  T = Eigen::Translation3d(cam_x, cam_y, cam_z);;
  
  // fix me: handle the situation when orientations are different.
  // load poses for two scans
  Eigen::Affine3d T_offset_gt;
  if (load_pos_1 && load_pos_2) {
    std::ifstream pos_f(pos_file_1);
    if(!pos_f.is_open()) {
      std::cerr<<"Error opening pos_1 file\n";
      return -1;
    }
    std::string line, scan_id;
    double scanner_x, scanner_y, scanner_z, axis_x, axis_y, axis_z, axis_angle;

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
    Eigen::Affine3d T1 =
    Eigen::Translation<double, 3>(scanner_x, scanner_y, scanner_z) *
    Eigen::AngleAxis<double>(axis_angle, axis);

    std::ifstream pos_f_2(pos_file_2);
    if(!pos_f_2.is_open()) {
      std::cerr<<"Error opening pos_2 file\n";
      return -1;
    }
    if(std::getline(pos_f_2, line)) {
      std::cerr<<"Read line: "<<line<<std::endl;
      std::istringstream iss(line);
      iss >> scan_id >> scanner_x >> scanner_y >> scanner_z
            >> axis_x >> axis_y >> axis_z >> axis_angle;
    } else {
      std::cerr<<"Error reading pos file\n";
      return -1;
    }
    pos_f_2.close();

    std::cerr<<"Scan ID: "<<scan_id<<std::endl;
    std::cerr<<"Scanner pose: "<<scanner_x<<", "<<scanner_y<<", "<<scanner_z<<std::endl;
    std::cerr<<"Axis: "<<axis_x<<", "<<axis_y<<", "<<axis_z<< ", " << axis_angle<< std::endl;

    axis = Eigen::Vector3d(axis_x, axis_y, axis_z);
    Eigen::Affine3d T2 =
    Eigen::Translation<double, 3>(scanner_x, scanner_y, scanner_z) *
    Eigen::AngleAxis<double>(axis_angle, axis);

    auto translation = T2.translation() - T1.translation();
    T_offset_gt = T1.inverse() * T2;

    std::cerr << "T1=\n"
              << T1.matrix() << "\n";
    std::cerr << "T2=\n"
              << T2.matrix() << "\n";
    std::cerr << "T_offset_gt=\n"
              << T_offset_gt.matrix() << "\n";
    // T_offset_gt = Eigen::Translation3d(T_offset_gt.translation())*
    // Eigen::AngleAxis<double>(M_PI*0/180., axis);

    // std::cerr << "T_offset_gt=\n"
    //           << T_offset_gt.matrix() << "\n";
  }
  else{
    T_offset_gt = Eigen::Translation<double, 3>(cam_x_gt, cam_y_gt, cam_z_gt) *
      Eigen::AngleAxis<double>(0, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxis<double>(0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxis<double>(0, Eigen::Vector3d::UnitZ());    
  }  

  // Filter cloud based on forward direction for better registration
  if (masking_distance > 0) {
    std::cout << "Applying masking with distance threshold: " << masking_distance << std::endl;

  Eigen::Vector3d forward_dir = -T_offset_gt.translation() / T_offset_gt.translation().norm();
  double distance_threshold = masking_distance;

  // Filter the second cloud if loaded
  if (load2ndPoints) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto& point : *cloud_2) {
      Eigen::Vector3d pt(point.x, point.y, point.z);
      double distance_forward = pt.dot(forward_dir);
      if (distance_forward < distance_threshold) {
        cloud_2_filtered->push_back(point);
      }
    }
    cloud_2 = cloud_2_filtered;
  }

  // std::cerr << "Filtered cloud size: " << cloud->size() << std::endl;
  // if (load2ndPoints) {
  //   std::cerr << "Filtered cloud_2 size: " << cloud_2->size() << std::endl;
  // }
  
  // // Save point clouds for visualization
  // std::string cloud_output_name = "cloud_filtered.pcd";
  // if (load2ndPoints) {
  //     // Combine both clouds and save together
  //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      
  //     // Color the clouds differently for distinction
  //     // colorPointCloud(cloud, 255, 0, 0);   // Color cloud 1 red
  //     // colorPointCloud(cloud_2, 0, 255, 0); // Color cloud 2 green
      
  //     *combined_cloud = *cloud + *cloud_2;
  //     // Apply voxel grid filtering to downsample the combined cloud
  //     pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
  //     float voxel_size = 0.1f; // 5cm voxel size
  //     voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
  //     voxel_grid.setInputCloud(combined_cloud);
  //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  //     voxel_grid.filter(*downsampled_cloud);

  //     std::cout << "Original combined cloud size: " << combined_cloud->size() << std::endl;
  //     std::cout << "Downsampled cloud size: " << downsampled_cloud->size() << std::endl;

  //     combined_cloud = downsampled_cloud;
  //     pcl::io::savePCDFile(cloud_output_name, *combined_cloud);
  //     std::cout << "Saved combined filtered clouds to " << cloud_output_name << std::endl;
  //   } else {
  //     pcl::io::savePCDFile(cloud_output_name, *cloud);
  //     std::cout << "Saved filtered cloud to " << cloud_output_name << std::endl;
  //   }
  }

  pcl::PointCloud<pcl::PointXYZ> labels;
  pcl::PointCloud<pcl::PointXYZ> labels_2;

  if (useLabel)
  {
    if (reader.read(label_file, labels) < 0)
    {
      std::cerr << "Error reading label file" << label_file << "\n";
      labels.is_dense = false;
      labels.points.clear();
    }
    else
    {
      std::cerr << "Read labels\n";
    }
  }

  if (useLabel_2)
  {
    if (reader.read(label_file_2, labels_2) < 0)
    {
      std::cerr << "Error reading label file" << label_file << "\n";
      labels_2.is_dense = false;
      labels_2.points.clear();
    }
    else
    {
      std::cerr << "Read labels\n";
    }
  }
  else
  {
    labels_2 = labels;
  }


  // ---------------------------------- first image ----------------------------------------//
  auto start_time = std::chrono::high_resolution_clock::now();
  auto start_time_after_loading_data = std::chrono::high_resolution_clock::now();
  
  std::cerr << "first image cam pose T=\n"
            << T.matrix() << "\n";

  std::shared_ptr<graph_matcher::CloudRenderer> rend(new graph_matcher::CloudRenderer(cp));
  rend->render(depth, intensity, *cloud, T);

  auto render_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> rendering_time = render_end_time - start_time_after_loading_data;
  std::cerr << "Time taken to render the first image: " << rendering_time.count() << " seconds\n";
  // color version of intensity image
  cv::Mat vis_converted, matches_display;
    if (show_img) {
    if (depthOnly) {
      cv::cvtColor(depth, vis_converted, cv::COLOR_GRAY2RGB);
    }
    else
    {
      cv::cvtColor(intensity, vis_converted, cv::COLOR_GRAY2RGB);
      // cv::imwrite("depth_1.png", depth);
      // cv::imwrite("intensity_1.png", intensity);
    }
  }
  std::vector<cv::Point> keypoints;
  Eigen::MatrixXd learned_features;  // fix me
  std::shared_ptr<graph_matcher::DetectorIfce> fd;
  if (useLabel)
  {
    fd = std::make_shared<graph_matcher::FakeDetector>(rend, T, labels);
  }
  else if (useModel)
  {

    auto start_loading_detector = std::chrono::high_resolution_clock::now();
    fd = std::make_shared<graph_matcher::FastrcnnDetector>(rend, model_file, useCPU);
    auto end_loading_detector = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> loading_detector_time = end_loading_detector - start_loading_detector;
    std::cerr << "Time taken to load the detector: " << loading_detector_time.count() << " seconds\n";
  } else {
    std::cerr << "No label provided, no model provided; Cannot find bolts" << std::endl;
  }

  auto start_detection_1 = std::chrono::high_resolution_clock::now();
  keypoints.clear();
  if (depthOnly)
  {
    cv::Mat zeroMat = cv::Mat::zeros(intensity.size(), intensity.type());
    fd->detect(zeroMat, depth, keypoints);   
  }
  else
  {
    fd->detect(intensity, depth, keypoints);
  }

  // TODO: for learned features
  // fd.getFeatures(learned_features);
  if (show_img) {
    drawKeypoints(vis_converted, keypoints);
  }
  graph_matcher::DescriptorExtractPtr extractor_first;
  extractor_first = getExtractor(rend, intensity, depth, feature_type);  


  auto end_detection_1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> detection_time_1 = end_detection_1 - start_detection_1;
  std::cerr << "Time taken to detect keypoints in first image: " << detection_time_1.count() << " seconds\n";
  matches_display = vis_converted.clone();


  auto start_time_graph_construction = std::chrono::high_resolution_clock::now();
  graph_matcher::GraphBuilder gb(rend);
  auto graph = gb.buildGraph(depth, keypoints, extractor_first);
  if (show_img) {
    gb.renderGraph(graph, vis_converted);  
  }  
  auto end_time_graph_construction = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> graph_construction_time = end_time_graph_construction - start_time_graph_construction;
  std::cerr << "Time taken to construct graph: " << graph_construction_time.count() << " seconds\n";

  if (isLoadingFeatures)
  { // fix me:, structure need to be reorganized
    graph->load(feature_file);
  } else {
    if (save_graph)
    {
      bool success_write = graph->save(input_1_name+".graph");
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr bolts_1, bolts_2;
  
  // cv::imshow("visualize image", vis_converted);

  if (batch_test) {
    // *************** batch_test ***********************
    int n_success=0;
    int n_skip = 0;
    double dx = x_bound/NX; 
    double dy = y_bound/NY; 
    double dyaw = yaw_bound/NYAW; 
    
    //sampling grid
    std::cerr << "Sampling grid: " << NX << " " << NY << " " << NYAW << std::endl;
    for (int nx=-NX/2; nx<(NX+1)/2; nx++) {
      for (int ny=-NY/2; ny<(NY+1)/2; ny++) {
        for (int nyaw=-NYAW/2; nyaw<(NYAW+1)/2; nyaw++) {

          //set transform
          double offset_x = nx*dx;
          double offset_y = ny*dy;
          double offset_yaw = nyaw*dyaw;
          Eigen::Affine3d deltaT = Eigen::Translation<double,3>(offset_x,offset_y,0)
                                  * Eigen::AngleAxis<double>(offset_yaw,Eigen::Vector3d::UnitZ());
          Eigen::Affine3d T_local = T * deltaT;
        
          Eigen::Affine3d deltaT_orig_frame = T_offset_gt;
          Eigen::Affine3d deltaT_frame = T_offset_gt*deltaT;
          // check if the offset makes view closer
          double shifted_distance = deltaT_frame.translation().norm();
          double orig_distance = deltaT_orig_frame.translation().norm();
          if (shifted_distance > orig_distance) {
            std::cerr << "Shifted distance is larger than original distance, skip\n";
            std::cerr << deltaT_frame.translation().transpose() << std::endl;
            ++n_skip;
            continue;
          }

          std::cerr << nx << " " << ny << " " << nyaw << std::endl;
          std::cerr << "x, y, yaw: " <<offset_x << ", " << offset_y << ", " << offset_yaw << std::endl;

          
          if (!load2ndPoints) 
          {      
            rend->render(depth, intensity, *cloud, T_local);
          }
          else 
          {
            rend->render(depth, intensity, *cloud_2, T_local);
          }             

          // color version of intensity image

          std::vector<cv::Point> keypoints_second;
          Eigen::MatrixXd learned_features;

          if (useLabel)
          {
            fd = std::make_shared<graph_matcher::FakeDetector>(rend, T, labels_2);
          }
          else if (!useModel)
          {
            std::cerr << "No label provided, no model provided; Cannot find bolts" << std::endl;
          }

          keypoints_second.clear();
          if (depthOnly)
          {
            cv::Mat zeroMat = cv::Mat::zeros(intensity.size(), intensity.type());
            fd->detect(zeroMat, depth, keypoints_second);   
          }
          else
          {
            fd->detect(intensity, depth, keypoints_second);
          }

          graph_matcher::DescriptorExtractPtr extractor_second;          
          extractor_second = getExtractor(rend, intensity, depth, feature_type);

          auto graph_second = gb.buildGraph(depth, keypoints_second, extractor_second);

          //------------------ now match the two ------------------//
          Eigen::Affine3d reg;
          std::vector<std::pair<int, int>> matches;
          
          matchGraphs(gb,
                      graph,
                      graph_second,
                      options,
                      feature_type,
                      do_connectivity,
                      matches,
                      reg);      
        
         
          auto reg_orig = deltaT * reg;
          std::cerr << "Got transform (first to second):\n" << reg.matrix() << std::endl;  
          std::cerr << "GT transform (first to second):\n" << deltaT_frame.inverse().matrix() << std::endl;
          std::cerr << "Got transform (first to second) in orig frame:\n" << reg_orig.matrix() << std::endl;
          std::cerr << "GT transform (first to second) in orig frame:\n" << deltaT_orig_frame.inverse().matrix() << std::endl;
          Eigen::Affine3d error = deltaT_orig_frame * reg_orig;
          Eigen::AngleAxisd alpha_error(error.rotation());
          double err_t = error.translation().norm();
          double err_r = alpha_error.angle() * 180.0 / M_PI;
          double gt_distance = deltaT_orig_frame.translation().norm();
          double err_dt = err_t / gt_distance;
          double err_dr = err_r / gt_distance;
          
          std::cerr << "GT distance=" << gt_distance << std::endl;
          std::cerr << "Off by t=" << err_t << " r=" << err_r << " degree" << std::endl;
          std::cerr << "Off by drift t=" << err_dt << " r=" << err_dr << " degree/m"<< std::endl;

          Logger::getInstance(log_file)->logOneTest(err_t, err_r, reg.matrix(),offset_x,offset_y,offset_yaw);
          if (err_t < 0.1 && err_r < 0.2)
          {
            n_success++;
          }
        }
      }
    }

    std::cerr << "Performed " << NX * NY * NYAW << " tests, with " << n_success << " converging, success rate is "
              << (double)n_success / ((NX * NY * NYAW)-n_skip) << std::endl;
  }
  else
  {
    // *************** registrate a pair ***********************
    // ---------------------------------- second image ----------------------------------------//
    // now let's do it for the second one as well
    // Eigen::Affine3d deltaT = Eigen::Affine3d::Identity() * Eigen::Translation<double, 3>(3, 0, -3);
    
    Eigen::Affine3d t_offset = Eigen::Affine3d::Identity();
    if(customized_offset) 
    {
      t_offset = Eigen::Translation<double, 3>(coffset_x, coffset_y, coffset_z)
              * Eigen::AngleAxis<double>(coffset_yaw,Eigen::Vector3d::UnitZ());
    }
    
    Eigen::Affine3d deltaT_orig_frame = Eigen::Affine3d::Identity() * T_offset_gt * t_offset;
    Eigen::Affine3d deltaT = T * deltaT_orig_frame;    
    Eigen::Affine3d T_total;


    if (!load2ndPoints) 
    {      
      T_total = deltaT;
      rend->render(depth, intensity, *cloud, T_total); 
      std::cerr << "second image (the same one with offset) cam pose T=\n"
              << T_total.matrix() << "\n";
    }
    else 
    {
      T_total = T * t_offset;
      rend->render(depth, intensity, *cloud_2, T_total);      
      std::cerr << "second image cam pose T=\n"
              << T_total.matrix() << "\n";
    }   
    // color version of intensity image
    if (depthOnly) 
    {
      cv::cvtColor(depth, vis_converted, cv::COLOR_GRAY2RGB);
    }
    else
    {
      cv::cvtColor(intensity, vis_converted, cv::COLOR_GRAY2RGB);
      // cv::imwrite("depth_2.png", depth);
      // cv::imwrite("intensity_2.png", intensity);
    }

    auto start_time_detection_2 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point> keypoints_second;
    Eigen::MatrixXd learned_features;
    if (useLabel)
    {
      fd = std::make_shared<graph_matcher::FakeDetector>(rend, T, labels_2);
    }
    else if (!useModel)
    {
      std::cerr << "No label provided, no model provided; Cannot find bolts" << std::endl;
    }
    
    keypoints_second.clear();
    if (depthOnly)
    {
      cv::Mat zeroMat = cv::Mat::zeros(intensity.size(), intensity.type());
      fd->detect(zeroMat, depth, keypoints_second);   
    }
    else
    {
      fd->detect(intensity, depth, keypoints_second);
    }

    

    if (show_img) {
      drawKeypoints(vis_converted, keypoints_second);
      cv::hconcat(matches_display, vis_converted, matches_display);
    }      

    graph_matcher::DescriptorExtractPtr extractor_second;    
    extractor_second = getExtractor(rend, intensity, depth, feature_type);

    std::chrono::duration<double> detection_time_2 = std::chrono::high_resolution_clock::now() - start_time_detection_2;
    std::cerr << "Detection time 2: " << detection_time_2 .count() << " seconds\n";

    auto graph_second = gb.buildGraph(depth, keypoints_second, extractor_second);
    if (show_img)
    {
      gb.renderGraph(graph_second, vis_converted);
      gb.renderGraph(graph, vis_converted, deltaT_orig_frame.inverse(), cv::Scalar(255, 0, 0));
    }
    
    
    
    if (save_graph)
    {
      bool success_write = graph_second->save(input_2_name + "_(" + input_1_name +").graph");
    }
    

    
    
    std::cout << "deltaT_orig_frame:\n" <<deltaT_orig_frame.matrix() << std::endl;
    std::cout << "deltaT:\n" <<deltaT.matrix() << std::endl;
    

    auto end_time_detection = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> detection_time = end_time_detection - start_time_after_loading_data;
    std::cerr << "Detection time all: " << detection_time.count() << " seconds\n";

    //------------------ now match the two ------------------//
    Eigen::Affine3d reg;
    std::vector<std::pair<int, int>> matches;
   
      matchGraphs(gb,
                  graph,
                  graph_second,
                  options,
                  feature_type,
                  do_connectivity,
                  matches,
                  reg);      
    

    auto end_time_registration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> registration_time = end_time_registration - end_time_detection;
    std::cerr << "Registration time: " << registration_time.count() << " seconds\n";

    if (show_img) {
      gb.renderGraph(graph, vis_converted, reg, cv::Scalar(100, 100, 0), 2);
    }
      
    // std::cerr << "Got transform (first to second):\n" << reg.matrix() << std::endl;
    auto reg_orig = T * reg;
    auto reg_final = t_offset*reg;
    auto deltaT_final = deltaT_orig_frame * (t_offset.inverse());
    Eigen::Affine3d error = deltaT_final * reg_final;
    std::cerr << "Transform (gt) in shifted point cloud frame\n" << deltaT_orig_frame.inverse().matrix() << std::endl;
    std::cerr << "Transform in shifted point cloud frame\n" << reg_orig.matrix() << std::endl;   
    std::cerr << "Transform (gt) in orig frame\n" << deltaT_final.inverse().matrix() << std::endl;
    std::cerr << "Transform in orig point cloud frame\n" << reg_final.matrix() << std::endl;
    Eigen::AngleAxisd alpha_error(error.rotation());

    double err_t = error.translation().norm();
    double err_r = alpha_error.angle() * 180.0 / M_PI;
    double gt_distance = deltaT_orig_frame.translation().norm();
    double err_dt = err_t / gt_distance;
    double err_dr = err_r / gt_distance;

    std::cerr << "GT distance=" << gt_distance << std::endl;
    std::cerr << "Off by t=" << err_t << " r=" << err_r << " degree" << std::endl;
    std::cerr << "Off by drift t=" << err_dt << " r=" << err_dr << " degree/m"<< std::endl;

    // Save reg_final.matrix() to a txt file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", now_tm);
    std::string reg_filename = "reg_matrix_(" + input_2_name + ")_(" + input_1_name + ")_" + timestamp + ".txt";

    std::string error_filename = "error_log_BaA_(" + input_2_name + ")_(" + input_1_name + ")_" + timestamp + ".json";
    writeTestResultToJson(err_t, err_r, err_dt, err_dr, error_filename, "BaA");

    std::ofstream reg_file(reg_filename);
    if (reg_file.is_open()) {
      reg_file << reg_final.matrix() << std::endl;
      reg_file.close();
      std::cout << "Transformation matrix saved to reg_final_matrix.txt" << std::endl;
    } else {
      std::cerr << "Unable to open file to save transformation matrix" << std::endl;
    }
    

    if (show_img) {

      cv::imshow("Graphs", vis_converted);
      cv::imshow("Matches", matches_display);
      cv::imshow("Depth Second", depth);

      cv::waitKey(0);
    }

    auto graph_reg_time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> graph_reg_time = graph_reg_time_end - start_time;
    std::cerr << "Graph registration time: " << graph_reg_time.count() << " seconds\n";
    if(options.refine_ICP) {
      auto start_time_icp = std::chrono::high_resolution_clock::now();
      std::cerr << "Refining with ICP\n";
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
      float grid_size = 0.1f;
      voxel_grid.setLeafSize(grid_size, grid_size, grid_size); // 0.1 cm = 0.001 m
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());
      voxel_grid.setInputCloud(cloud);
      voxel_grid.filter(*cloud_downsampled);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());
      voxel_grid.setInputCloud(cloud_2);
      voxel_grid.filter(*cloud_2_downsampled);
      
      std::cerr << "Downsampled cloud size: " << cloud_downsampled->size() << std::endl;
      std::cerr << "Downsampled cloud_2 size: " << cloud_2_downsampled->size() << std::endl;
      pcl::transformPointCloud(*cloud_downsampled, *cloud_downsampled, reg.matrix());
      
      // Remove points within a 10 meter cylinder (radius = 10m, along Z axis, centered at origin)
      auto removeCylinder = [&reg](pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float radius, bool use_offset = false) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        filtered->reserve(cloud->size());
        auto center = reg.translation();
        for (const auto& pt : *cloud) {
          float delta_x, delta_y;
          if (use_offset) {
            delta_x = pt.x - center.x();
            delta_y = pt.y - center.y();
          } else {
            delta_x = pt.x;
            delta_y = pt.y;
          }
          
          float dist_xy = std::sqrt(delta_x * delta_x + delta_y * delta_y);
          if (dist_xy > radius) {
            filtered->push_back(pt);
          }
        }
        cloud.swap(filtered);
      };
      // removeCylinder(cloud_downsampled, 12.0f, true);
      // removeCylinder(cloud_2_downsampled, 12.0f, false);
      
      std::cerr << "removeCylinder cloud size: " << cloud_downsampled->size() << std::endl;
      std::cerr << "removeCylinder cloud_2 size: " << cloud_2_downsampled->size() << std::endl;

      // Estimate normals for target cloud
      pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
      pcl::NormalEstimation<PointT, pcl::Normal> ne;
      ne.setInputCloud(cloud_2_downsampled);
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      ne.setSearchMethod(tree);
      ne.setKSearch(20);
      ne.compute(*normals);

      // Combine XYZ and normals into PointNormal
      pcl::PointCloud<PointNormalT>::Ptr cloud_out_with_normals(new pcl::PointCloud<PointNormalT>);
      pcl::concatenateFields(*cloud_2_downsampled, *normals, *cloud_out_with_normals);

      // Convert input cloud to PointNormal as well (no real normals needed here)
      pcl::PointCloud<PointNormalT>::Ptr cloud_in_with_normals(new pcl::PointCloud<PointNormalT>);
      pcl::copyPointCloud(*cloud_downsampled, *cloud_in_with_normals);

      // for icp p2p, we can just use the downsampled cloud without normals
      
      // pcl::IterativeClosestPoint<PointNormalT, PointNormalT> icp;
      // pcl::PointCloud<PointNormalT> Final;
      // icp.setInputSource(cloud_in_with_normals);
      // icp.setInputTarget(cloud_out_with_normals);          
      
      
      pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
      pcl::PointCloud<pcl::PointXYZRGB> Final;
      icp.setInputSource(cloud_downsampled);
      icp.setInputTarget(cloud_2_downsampled);              

      // icp.setMaximumIterations(5000);
      icp.setMaximumIterations(50);
      icp.setMaxCorrespondenceDistance(1.5*grid_size);
      icp.setTransformationEpsilon(1e-8);
      icp.setEuclideanFitnessEpsilon(1e-8);
      icp.setUseReciprocalCorrespondences(true);
      // icp.setTransformationEstimation(
      //   pcl::make_shared<pcl::registration::TransformationEstimationPointToPlane<PointNormalT, PointNormalT>>());
      icp.align(Final);
      
      auto end_time_icp = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> icp_time = end_time_icp - start_time_icp;
      std::cerr << "ICP time: " << icp_time.count() << " seconds" << std::endl;
      std::cerr << "ICP converged: " << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
      Eigen::Matrix4d icp_transform_d = icp_transform.cast<double>();
      reg = Eigen::Affine3d(icp_transform_d) * reg;
      std::cerr << "Final transformation matrix:\n" << reg.matrix() << std::endl;
      // Save reg_final.matrix() to a txt file
      std::string reg_icp_filename = "reg_matrix_icp_(" + input_2_name + ")_(" + input_1_name + ")_" + timestamp + ".txt";
      std::ofstream reg_file(reg_icp_filename);
      if (reg_file.is_open()) {
        reg_file << reg.matrix() << std::endl;
        reg_file.close();
        std::cout << "Transformation matrix saved to reg_final_matrix.txt" << std::endl;
      } else {
        std::cerr << "Unable to open file to save transformation matrix" << std::endl;
      }

      
      Eigen::Affine3d error_icp;
      error_icp.matrix() = deltaT_final * reg.matrix();
      Eigen::AngleAxisd alpha_error_icp(error_icp.rotation());
      double err_t = error_icp.translation().norm();
      double err_r = alpha_error_icp.angle() * 180.0 / M_PI;
      double gt_distance = deltaT_orig_frame.translation().norm();
      double err_dt = err_t / gt_distance;
      double err_dr = err_r / gt_distance;

      std::cerr << "GT distance=" << gt_distance << std::endl;
      std::cerr << "Off by t=" << err_t << " r=" << err_r << " degree" << std::endl;
      std::cerr << "Off by drift t=" << err_dt << " r=" << err_dr << " degree/m"<< std::endl;

      std::string error_filename = "error_log_BaA+ICP_(" + input_2_name + ")_(" + input_1_name + ")_" + timestamp + ".json";
      writeTestResultToJson(err_t, err_r, err_dt, err_dr, error_filename, "BaA+ICP");

      auto total_time_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> total_time = total_time_end - start_time_after_loading_data;
      std::string time_filename = "time_log_BaA+ICP_(" + input_2_name + ")_(" + input_1_name + ")_" + timestamp + ".json";
      writeTestTimeCostToJson(detection_time.count(), registration_time.count(), icp_time.count(), total_time.count(), time_filename, "BaA+ICP");
      std::cerr << "Total time: " << total_time.count() << " seconds" << std::endl;
    }
    // std::cerr << "T.inverse reg\n" << reg_orig.matrix() << std::endl;    
    // connect two point cloud
    if (connect_pts) {

      std::cerr << "start stitching two point coulds..." << std::endl;
      // Apply the transformation matrix to the second point cloud
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1_transformed_gt(new pcl::PointCloud<pcl::PointXYZRGB>);
      // Eigen::Matrix4d transform_matrix = reg.matrix().cast<float>();
      Eigen::Matrix4d transform_matrix_gt = deltaT_orig_frame.inverse().matrix();
      pcl::transformPointCloud(*cloud, *cloud_1_transformed_gt, transform_matrix_gt);
      
      Eigen::Matrix4d transform_matrix = reg_final.matrix();
      pcl::transformPointCloud(*cloud, *cloud_1_transformed, transform_matrix);
      
      // Combine the two point clouds      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combined(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_combined_gt(new pcl::PointCloud<pcl::PointXYZRGB>);

      // assign different color to different cloud      
      colorPointCloud(cloud_1_transformed, 255, 0, 0);   // Color cloud 1 red
      colorPointCloud(cloud_1_transformed_gt, 255, 0, 0);   // Color cloud 1 red
      colorPointCloud(cloud_2, 128, 128, 128); // Color cloud 2 gray
      
      // combine cloud
      *cloud_combined = *cloud_2 + *cloud_1_transformed;
      *cloud_combined_gt = *cloud_2 + *cloud_1_transformed_gt;

      if (pcl::io::savePCDFileBinary("cloud_combined_gt.pcd", *cloud_combined_gt) == -1)
      {
          PCL_ERROR("Couldn't save file cloud_combined_gt.pcd \n");
          return (-1);
      }
      if (pcl::io::savePCDFileBinary("cloud_combined.pcd", *cloud_combined) == -1)
      {
          PCL_ERROR("Couldn't save file cloud_combined.pcd \n");
          return (-1);
      }
      std::cout << "registered cloud saved! (cloud_combined.pcd)" << std::endl;
    }
    // draw matches
    for (auto itr = matches.begin(); itr != matches.end(); itr++)
    {
      cv::Point ptA, ptB;
      ptA.x = keypoints[itr->first].y;
      ptA.y = keypoints[itr->first].x;
      ptB.x = keypoints_second[itr->second].y + cp.image_y;
      ptB.y = keypoints_second[itr->second].x;

      cv::line(matches_display, ptA, ptB, cv::Scalar(0.7, 0.1, 0.1), 2);
    }

  }

  return 0;
}
