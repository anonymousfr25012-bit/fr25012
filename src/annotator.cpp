#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <boost/program_options.hpp> 
#include <boost/make_shared.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Geometry>

#include <sstream>      // std::stringstream
#include <filesystem>

#include <boomer2_tools/CloudRenderer.hh>

namespace po = boost::program_options;
namespace fs = std::filesystem;

std::vector<cv::Point> clicked_points;
std::vector<cv::Point> un_clicked_points;

// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse( int event, int x, int y, int, void* )
{
 switch( event )
 {
   case cv::EVENT_LBUTTONDOWN:
     //add clicked point to list
     clicked_points.push_back(cv::Point(y,x));
     break;
   case cv::EVENT_RBUTTONDOWN:
     //add clicked point to list
     un_clicked_points.push_back(cv::Point(y,x));
     break;
 }
}

//finds any label points that are close to unclicked points and deletes them, deletes clicked points as well
void processUnClickedPoints(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ> &labels, graph_matcher::CamParams &cp, Eigen::Affine3d &T) {
  
  graph_matcher::CloudRenderer rend(cp);
  for(int p=0; p<un_clicked_points.size(); p++) {

    cv::Point ul,lr,unclicked;
    unclicked.x=un_clicked_points[p].x;
    unclicked.y=un_clicked_points[p].y;
    //check if we are removing a clicked point
    int del=-1;
    for(int q=0; q<clicked_points.size(); q++) {
      if(abs(unclicked.x-clicked_points[q].x) < 2 && abs(unclicked.y-clicked_points[q].y) < 2) {
        del=q;
        break;
      }
    }

    if(del>=0) {
      clicked_points.erase(clicked_points.begin()+del);
      continue;
    }

    //we have to do the tougher job of erasing a label point instead
    ul.x = MAX(unclicked.x-5,0);
    ul.y = MAX(unclicked.y-5,0);
    lr.x = MIN(unclicked.x+5,cp.image_x-1);
    lr.y = MIN(unclicked.y+5,cp.image_y-1);

    float distance=0;
    int n_hood=0;
    float d;
    for(int i=ul.x; i<=lr.x; i++) {
      for(int j=ul.y; j<=lr.y; j++) {
        d=depth.at<float>(i,j);
        if(d<1.0) {
          distance+=d;
          n_hood++;
        }
      }
    }
    //normalize by number of pixels in hood
    distance = (distance/n_hood)*cp.max_dist;

    //backproject
    Eigen::Vector3d pv,pv_trans;
    rend.backproject(unclicked, pv);
    pv = pv*distance;
    //pv<<(unclicked.x - cp.center_x)*distance/cp.f,
    //    (unclicked.y - cp.center_y)*distance/cp.f,
    //    distance;
    
    //transform to global
    pv_trans = T*pv;

    //search through labels and remove
    ;
    for(auto itr = labels.begin();itr!=labels.end(); ) {
      pv<<itr->x,itr->y,itr->z;
      if((pv-pv_trans).norm()<0.05) {
        labels.erase(itr);
        break;
      }
      else itr++;
    }
  }
  un_clicked_points.clear();
}

void processClickedPoints(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ> &labels, graph_matcher::CamParams &cp, Eigen::Affine3d &T) {
  
  graph_matcher::CloudRenderer rend(cp);
  for(int p=0; p<clicked_points.size(); p++) {
    cv::Point ul,lr,clicked;
    clicked.x=clicked_points[p].x;
    clicked.y=clicked_points[p].y;

    ul.x = MAX(clicked.x-5,0);
    ul.y = MAX(clicked.y-5,0);
    lr.x = MIN(clicked.x+5,cp.image_x-1);
    lr.y = MIN(clicked.y+5,cp.image_y-1);

    float distance=0;
    int n_hood=0;
    float d;
    for(int i=ul.x; i<=lr.x; i++) {
      for(int j=ul.y; j<=lr.y; j++) {
        d=depth.at<float>(i,j);
        if(d<1.0) {
          distance+=d;
          n_hood++;
        }
      }
    }
    //normalize by number of pixels in hood
    distance = (distance/n_hood)*cp.max_dist;

    //backproject
    Eigen::Vector3d pv,pv_trans,pv_back;
    rend.backproject(clicked, pv);
    pv = pv*distance;
    //pv<<(clicked.x - cp.center_x)*distance/cp.f,
    //    (clicked.y - cp.center_y)*distance/cp.f,
    //    distance;
    //transform to global
    //pv = T.rotation()*pv+T.translation();
    pv_trans = T*pv;
    pv_back = pv - T.rotation().transpose()*(pv_trans-T.translation());

    //std::cerr<<"depth is "<<distance<<" reprojection error is "<<pv_back.norm()<<std::endl;

    pcl::PointXYZ pt;
    pt.x=pv_trans(0); pt.y=pv_trans(1); pt.z=pv_trans(2);
    labels.push_back(pt);
  }
  clicked_points.clear();
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

int main(int argc, char** argv) {

  std::string in_file, label_file, base_name, pt_type;
  double cam_x, cam_y, cam_z;
  int image_x, image_y;
  int NX,NY,NZ,NYAW,NPITCH,NROLL;
  char yaw_axis;
  double minor_angles_bound = 0.3;
  double xy_bound = 3;
  double z_bound = 2;

  double grid_resolution;

  po::options_description desc("Allowed options");
  desc.add_options() 	
    ("help", "produce help message") 	
    ("annotate", "runs the annotation GUI and saves the 3D points") 	
    ("generate", "runs in data generation mode and saves pngs and labels") 	
    ("fisheye", "render a fisheye image") 	
    ("depth_only", "render depth images only") 	
    ("pt_type", po::value<std::string>(&pt_type)->default_value("XYZRGB"), "point cloud type, XYZRGB or XYZI, (or XYZ)")
    ("base_name", po::value<std::string>(&base_name), "prefix to all generated file names") 	
    ("input", po::value<std::string>(&in_file), "name of the input pcd file") 	
    ("labels", po::value<std::string>(&label_file), "name of the pcd file that holds labeled points") 	
    ("resolution", po::value<double>(&grid_resolution)->default_value(0.1), "size of the grid to use")
    ("cam_x", po::value<double>(&cam_x)->default_value(0.0), "X initial position of camera")
    ("cam_y", po::value<double>(&cam_y)->default_value(0.0), "Y initial position of camera")
    ("cam_z", po::value<double>(&cam_z)->default_value(0.0), "Z initial position of camera")
    ("image_x", po::value<int>(&image_x)->default_value(500), "Image height to render")
    ("image_y", po::value<int>(&image_y)->default_value(500), "Image width to render")
    ("num_x", po::value<int>(&NX)->default_value(10), "Number of samples around X (generate only)")
    ("num_y", po::value<int>(&NY)->default_value(10), "Number of samples around Y (generate only)")
    ("num_z", po::value<int>(&NZ)->default_value(3), "Number of samples around Z (generate only)")
    ("num_roll", po::value<int>(&NROLL)->default_value(3), "Number of samples around roll (generate only)")
    ("num_pitch", po::value<int>(&NPITCH)->default_value(3), "Number of samples around pitch (generate only)")
    ("num_yaw", po::value<int>(&NYAW)->default_value(10), "Number of samples around yaw (generate only)")
    ("angles_bound", po::value<double>(&minor_angles_bound)->default_value(0.3), "minor_angles_bound around x and y axis")
  ;

  po::variables_map vm;     
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))     
  { 	
    std::cout << desc << "\n"; 	
    return 1;     
  }
  bool annotate = vm.count("annotate");
  bool generate = vm.count("generate");
  bool fisheye = vm.count("fisheye");
  bool depth_only = vm.count("depth_only");
  
  if((!annotate&&!generate) || (annotate&&generate)) {
    std::cout<<"Select either annotation or label generation mode, but not both\n";
    return -1;
  }
  if (!vm.count("input") || !vm.count("labels"))     
  {  
    std::cout << "Please provide an input and label file name tags. Run with --help for usage.\n";	
    return -1;     
  }
  if(generate && !vm.count("base_name")) {
    std::cerr<<"Base name required for generating data\n";
    return -1;
  }

//  pcl::PointCloud<pcl::PointXYZI> cloud;


pcl::PCDReader reader;

pcl::PointCloud<pcl::PointXYZRGB> cloud;
if(pt_type == "XYZI") {

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
  if(reader.read(in_file, *cloud_xyzi) < 0) {
    std::cerr<<"Error reading pcd file\n";
    return -1;
  }

  // Convert to XYZRGB
  cloud = *convertXYZIToXYZRGB(cloud_xyzi);
}
else {
  if(reader.read(in_file, cloud) < 0) {
    std::cerr<<"Error reading pcd file\n";
    return -1;
  }
}

  std::cerr << "PointCloud read: " << cloud.width * cloud.height
    << " data points \n";

  graph_matcher::CamParams cp;
  cp.center_x = image_x/2.0;
  cp.center_y = image_y/2.0;
  cp.f = 250.0;
  cp.image_x = image_x;
  cp.image_y = image_y;
  cp.max_dist = 15;
  cp.isFisheye = fisheye;

  if(fisheye) {
    std::cerr << "render fisheye images" << std::endl;
    cp.f = 350.0;
  }
   
  if(annotate) {

    cv::namedWindow ("Depth", cv::WINDOW_AUTOSIZE);
    cv::namedWindow ("Intensity", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Intensity", onMouse, 0);

    cv::Mat depth = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(1.0));
    cv::Mat intensity = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(0.0));

    bool quit=false;

    Eigen::Affine3d T =  
      Eigen::Translation<double,3>(cam_x,cam_y,cam_z)*
      Eigen::AngleAxis<double>(0,Eigen::Vector3d::UnitZ()) ;

    pcl::PointCloud<pcl::PointXYZ> labels;
    if(reader.read(label_file, labels) < 0) {
      std::cerr<<"Error reading label file"<<label_file<<"\n";
      labels.is_dense=false;
      labels.points.clear();
    } else {
      std::cerr<<"Read labels\n";
    } 
    
    graph_matcher::CloudRenderer rend(cp);

    while(!quit) {  

      std::cerr<<"cam pose T=\n"<<T.matrix()<<"\n";


      
      rend.render(depth,intensity,cloud,T);
      //rend.render(depth,intensity,cloud,T);
      //time to show our work
      cv::Mat depth_converted;
      depth.convertTo(depth_converted, CV_8UC1, 255.);
      // Holds the colormap version of the image:
      double minVal, maxVal;
      cv::minMaxLoc(depth, &minVal, &maxVal);

      std::cout << "Depth min: " << minVal << "  max: " << maxVal << std::endl; 
      cv::minMaxLoc(depth_converted, &minVal, &maxVal);

      std::cout << "Depth min: " << minVal << "  max: " << maxVal << std::endl; 
      cv::Mat img_color;
      // Apply the colormap:
      cv::applyColorMap(depth_converted, img_color, cv::COLORMAP_JET);

      //color version of intensity image
      cv::Mat intensity_converted;
      cv::cvtColor(intensity, intensity_converted, cv::COLOR_GRAY2RGB);
      clicked_points.clear();

      char key=-1;

      while(key == -1) {

        //check if we need to re-draw or process the de-selected points
        if(un_clicked_points.size()>0) {
          cv::cvtColor(intensity, intensity_converted, cv::COLOR_GRAY2RGB);
          processUnClickedPoints(depth, labels, cp, T);
        }

        //draw current clicked points
        for(int p=0; p<clicked_points.size(); p++) {
          cv::Point ul,lr;
          ul.y = MAX(clicked_points[p].x-5,0);
          ul.x = MAX(clicked_points[p].y-5,0);
          lr.y = MIN(clicked_points[p].x+5,cp.image_x-1);
          lr.x = MIN(clicked_points[p].y+5,cp.image_y-1);
          cv::rectangle(intensity_converted,ul,lr,
              cv::Scalar(255,0,0), 2, cv::LINE_8);
        }

        //project and draw label points
        for(int p=0; p<labels.points.size(); p++) {
          Eigen::Vector3d point, point_in_cam;
          point<<labels.points[p].x,labels.points[p].y,labels.points[p].z;
          //calculate point coordinates relative to camera
          point_in_cam = T.rotation().transpose()*(point-T.translation());
          //std::cerr<<"Point = "<<point.transpose()<<" in cam = "<<point_in_cam.transpose()<<std::endl;

          double depth_threshold = 10;  // adjust the largest depth for labeling

          if(point_in_cam(2) <= 0.0|| point_in_cam.norm() > depth_threshold) {
            //n_behind++;
            continue; //point behind camera plane
          }

          cv::Point pixel;
          rend.project(pixel,point_in_cam);
          int x = pixel.x, y=pixel.y;
          //int x = (cp.f*point_in_cam(0))/point_in_cam(2) + cp.center_x;
          //int y = (cp.f*point_in_cam(1))/point_in_cam(2) + cp.center_y;

          //check if in FOV
          if(x<0 || x >= cp.image_x ||
              y<0 || y >= cp.image_y ) {
            continue;
          }
          //std::cerr<<"pixel "<<x<<" "<<y<<std::endl;

          cv::Point ul,lr;
          ul.y = MAX(x-5,0);
          ul.x = MAX(y-5,0);
          lr.y = MIN(x+5,cp.image_x-1);
          lr.x = MIN(y+5,cp.image_y-1);
          cv::rectangle(intensity_converted,ul,lr,
              cv::Scalar(0,255,0), 2, cv::LINE_8);
        }

        cv::imshow("Depth", img_color);
        cv::imshow("Intensity", intensity_converted);
        
        key = cv::waitKey(10);
        if(key == -1) continue;
        
        cv::Mat intensity_norm, intensity_rgb;
        switch(key) 
        {
          case 'p': 
            quit=true; 

            cv::imwrite("depth_colored.png", img_color);
            cv::normalize(intensity, intensity_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::cvtColor(intensity_norm, intensity_rgb, cv::COLOR_GRAY2BGR);
            cv::imwrite("intensity.png", intensity_rgb);          
            break;

          case 'w':
            T = T*Eigen::Translation<double,3>(0,0,0.5);
            break;
          case 's':
            T = T*Eigen::Translation<double,3>(0,0,-0.5);
            break;
          case 'v':
            T = T*Eigen::Translation<double,3>(0,-0.5,0);
            break;
          case 'b':
            T = T*Eigen::Translation<double,3>(0,0.5,0);
            break;
          case 'f':
            T = T*Eigen::Translation<double,3>(0.5,0,0);
            break;
          case 'g':
            T = T*Eigen::Translation<double,3>(-0.5,0,0);
            break;
          case 'a':
            T = T*Eigen::AngleAxis<double>(0.5, Eigen::Vector3d::UnitX());
            break;
          case 'd':
            T = T*Eigen::AngleAxis<double>(-0.5, Eigen::Vector3d::UnitX());
            break;
          case 'q':
            T = T*Eigen::AngleAxis<double>(-0.5, Eigen::Vector3d::UnitY());
            break;
          case 'e':
            T = T*Eigen::AngleAxis<double>(0.5, Eigen::Vector3d::UnitY());
            break;
          case 'r':
            T = T*Eigen::AngleAxis<double>(-0.5, Eigen::Vector3d::UnitZ());
            break;
          case 't':
            T = T*Eigen::AngleAxis<double>(0.5, Eigen::Vector3d::UnitZ());
            break;
          case 'i':
            T = 
              Eigen::Translation<double,3>(cam_x,cam_y,cam_z)*
              Eigen::AngleAxis<double>(0,Eigen::Vector3d::UnitZ()) ;
            break;
          case 'o':
            processClickedPoints(depth, labels, cp, T);
            break;
          default:
            std::cout<<"Controls:\n\n";
            std::cout<<"\tw:\tmove forward\n";
            std::cout<<"\ts:\tmove back\n";
            std::cout<<"\ta:\tpan left\n";
            std::cout<<"\td:\tpan right\n";
            std::cout<<"\tq:\tpitch up\n";
            std::cout<<"\te:\tpitch down\n";
            std::cout<<"\tr/t:\troll around camera axis\n";
            std::cout<<"\tf/g:\t move up/down\n";
            std::cout<<"\tv/b:\t move left/right\n";
            std::cout<<"\n\ti:\treset camera\n";
            std::cout<<"\n\to:\tadd the selected labels\n";
            std::cout<<"\tp:\tquit\n\n";
            break;
        }
      }
    }

    std::cerr<<"Saving labels to file "<<label_file<<std::endl;
    //save the label points
    pcl::PCDWriter writer;
    writer.write (label_file, labels);

  }

  if(generate) {
    //here load labels and generate data
    pcl::PointCloud<pcl::PointXYZ> labels;
    if(reader.read(label_file, labels) < 0) {
      std::cerr<<"Error reading label file"<<label_file<<"\n";
      return -1;
    }
   
    
    graph_matcher::CloudRenderer rend(cp);
    cv::Mat depth = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(1.0));
    cv::Mat intensity = cv::Mat(image_x, image_y, CV_32FC1, cv::Scalar(0.0));
    double yaw_bound = 2*M_PI;
    double dx = xy_bound/NX; 
    double dy = xy_bound/NY; 
    double dz = z_bound/NZ; 
    double droll = minor_angles_bound/NROLL; 
    double dpitch = minor_angles_bound/NPITCH; 
    double dyaw = yaw_bound/NYAW; 
    int seq_num=0;
    //sampling grid

    // create folder
    std::string fname_data;

    fname_data = std::string("data/");
    if (fs::create_directory(fname_data)) {
      std::cout << "Folder " << fname_data.c_str() << " created successfully."  << std::endl;
    } else {
      std::cerr << "Failed to create folder." << std::endl;
      // return -1;
    }

    fs::create_directory("data/Labels/");
    fs::create_directory("data/DepthImages/");
    fs::create_directory("data/IntensityImages/");

    
    for (int nx=-NX/2; nx<(NX+1)/2; nx++) {
    for (int ny=-NY/2; ny<(NY+1)/2; ny++) {
    for (int nz=-NZ/2; nz<(NZ+1)/2; nz++) {
    for (int nroll=-NROLL/2; nroll<(NROLL+1)/2; nroll++) {
    for (int npitch=-NPITCH/2; npitch<(NPITCH+1)/2; npitch++) {
    for (int nyaw=-NYAW/2; nyaw<(NYAW+1)/2; nyaw++) {
      //set transform

      Eigen::Vector3d x_axis, y_axis, z_axis;
        x_axis = Eigen::Vector3d::UnitX();
        y_axis = Eigen::Vector3d::UnitY();
        z_axis = Eigen::Vector3d::UnitZ();
      
      
      Eigen::Affine3d T = Eigen::Translation<double,3>(cam_x+nx*dx,cam_y+ny*dy,cam_z+nz*dz)*
        Eigen::AngleAxis<double>(nroll*droll,x_axis) *
        Eigen::AngleAxis<double>(npitch*dpitch,y_axis) *
        Eigen::AngleAxis<double>(nyaw*dyaw,z_axis);
        

      std::cerr<<nx<<" "<<ny<<" "<<nz<<" "<<nroll<<" "<<npitch<<" "<<nyaw<<std::endl;
      std::cerr<<"cam pose T=\n"<<T.matrix()<<"\n";
      
      //graph_matcher::CloudRenderer::render(depth,intensity,cloud,cp,T);
      if (depth_only) {
        rend.renderDepth(depth,cloud,T);
      } 
      else
      {
	std::cerr <<"depth and intensity " << std::endl;
        rend.render(depth,intensity,cloud,T);
      }
      

      //time to show our work
      cv::Mat depth_converted;
      depth.convertTo(depth_converted, CV_8UC1, 255.);
      // Holds the colormap version of the image:
      cv::Mat img_color;
      // Apply the colormap:
      cv::applyColorMap(depth_converted, img_color, cv::COLORMAP_JET);

      //color version of intensity image
      cv::Mat intensity_converted;
      if(!depth_only) {
        
      cv::cvtColor(intensity, intensity_converted, cv::COLOR_GRAY2RGB); 
      }
      clicked_points.clear();

      std::vector<cv::Point> visible;
      std::vector<int> visible_ids;
      std::vector<pcl::PointXYZ> visible_3d;

      //project and draw label points
      for(int p=0; p<labels.points.size(); p++) {
        Eigen::Vector3d point, point_in_cam;
        point<<labels.points[p].x,labels.points[p].y,labels.points[p].z;
        //calculate point coordinates relative to camera
        point_in_cam = T.rotation().transpose()*(point-T.translation());
        //std::cerr<<"Point = "<<point.transpose()<<" in cam = "<<point_in_cam.transpose()<<std::endl;

        double depth_threshold = 8;  // adjust the largest depth for labeling

        if(point_in_cam(2) <= 0.0|| point_in_cam.norm() > depth_threshold) {
          //n_behind++;
          continue; //point behind camera plane
        }

        cv::Point pixel;
        rend.project(pixel,point_in_cam);
        int x = pixel.x, y=pixel.y;
        //int x = (cp.f*point_in_cam(0))/point_in_cam(2) + cp.center_x;
        //int y = (cp.f*point_in_cam(1))/point_in_cam(2) + cp.center_y;

        //check if in FOV
        if(x<0 || x >= cp.image_x ||
            y<0 || y >= cp.image_y ) {
          continue;
        }
        //std::cerr<<"pixel "<<x<<" "<<y<<std::endl;

        cv::Point ul,lr;
        ul.y = MAX(x-5,0);
        ul.x = MAX(y-5,0);
        lr.y = MIN(x+5,cp.image_x-1);
        lr.x = MIN(y+5,cp.image_y-1);

        if(depth_only) {
            cv::rectangle(img_color,ul,lr,
            cv::Scalar(0,255,0), 2, cv::LINE_8);
        }
        else {
          cv::rectangle(intensity_converted,ul,lr,
              cv::Scalar(0,255,0), 2, cv::LINE_8);
        }
        visible.push_back(ul);
        visible.push_back(lr);
        visible_ids.push_back(p);
        visible_3d.push_back(labels.points[p]);
      }

      cv::imshow("Depth", img_color);
      if (!depth_only) {
        cv::imshow("Intensity", intensity_converted);
      }
      

      cv::Mat depth_save, intensity_save;
      depth.convertTo(depth_save, CV_16UC1, cp.max_dist*1000.); //depth in milimeters

      if (!depth_only){
        intensity.convertTo(intensity_save, CV_16UC1, 1000.); //intensity in *1000 raw
      }
      

      std::stringstream ss;
      std::string fname;
      // create folder if there is none
      
      ss<<"data/Labels/"<<base_name<<"_label_"<<seq_num<<".csv";
      ss>>fname;
      std::ofstream logfile(fname);
      ss.clear();
      
      /*******************************************************************
        *
        * Log file format:
        * Image_ID translation_x translation_y translation_z euler_r euler_p euler_y num_bolts
        * [bbox_ul_x bbox_ul_y bbox_lr_x bbox_lr_y]_N [unique_id]_N [bolt_x bolt_y bolt_z]_N
        *
       ******************************************************************/

      Eigen::Vector3d t=T.translation(), r=T.rotation().eulerAngles(0,1,2);
      logfile<<seq_num<<" "<<t(0)<<" "<<t(1)<<" "<<t(2)<<" "
            <<r(0)<<" "<<r(1)<<" "<<r(2)<<" "
            <<visible.size()/2<<" ";
      for(int v=0; v<visible.size()-1; v+=2) {
        logfile<<visible[v].x<<" "<<visible[v].y<<" "<<visible[v+1].x<<" "<<visible[v+1].y<<" ";
      }
      for(int v=0; v<visible_ids.size(); v++) {
        logfile<<visible_ids[v]<<" ";
      }
      for(int v=0; v<visible_3d.size(); v++) {
        logfile<<visible_3d[v].x<<" "<<visible_3d[v].y<<" "<<visible_3d[v].z<<" ";
      }
      logfile<<std::endl;

      ss<<"data/DepthImages/"<<base_name<<"_depth_"<<seq_num<<".png";
      ss>>fname;
      cv::imwrite(fname, depth_save);
      ss.clear();

      if(!depth_only) {

        ss<<"data/IntensityImages/"<<base_name<<"_intensity_"<<seq_num<<".png";
        ss>>fname;
        cv::imwrite(fname, intensity_save);
        ss.clear();
      }
#if 0
      ss<<base_name<<"_labled_"<<seq_num<<".jpeg";
      ss>>fname;
      cv::imwrite(fname, intensity_converted);
#endif
      seq_num++;
      
      cv::waitKey(10);

    }
    }
    }
    }
    }
    }
    /*
       const cv::Mat proto = cv::Mat(image_x, image_y, CV_32SC1, max_dist*1000);
       std::vector<cv::Mat> channels;
       for (int i = 0; i < 3; i++)
       channels.push_back(proto);
       cv::Mat image;
       cv::merge(channels, image);
     */


  }
  return 0;
}
