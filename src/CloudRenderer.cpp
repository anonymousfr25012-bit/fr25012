#include <boomer2_tools/CloudRenderer.hh>

#include <omp.h>

using namespace graph_matcher;

//void CloudRenderer::render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZI> &cloud, CamParams &cp, Eigen::Affine3d &T) {
void CloudRenderer::render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZRGB> &cloud, CamParams &cp, Eigen::Affine3d &T) {
  
    depth.create(cp.image_x, cp.image_y, CV_32FC1);
    intensity.create(cp.image_x, cp.image_y, CV_32FC1);
    depth = cv::Scalar(1.0);
    intensity = cv::Scalar(0.0);  
    size_t imsize = cp.image_x*cp.image_y;
    #pragma omp parallel shared(depth) shared(intensity) shared(cloud) num_threads(10)
    {

    float *depth_private = new float[imsize];
    for(int q=0; q<imsize; q++) depth_private[q] = 1.0;
    float *intensity_private = new float[imsize];
    for(int q=0; q<imsize; q++) intensity_private[q] = 0.0;

    std::cerr<< "OpenMP running from process "<<omp_get_thread_num()<<std::endl;
    Eigen::Vector3d point, point_in_cam;
    
    #pragma omp for
    for(int i=0; i<cloud.points.size(); i++) {
      point<<cloud.points[i].x,cloud.points[i].y,cloud.points[i].z;

      //calculate point coordinates relative to camera
      point_in_cam = T.rotation().transpose()*(point-T.translation());

      if(point_in_cam(2) <= 0.0) {
        //n_behind++;
        continue; //point behind camera plane
      }

      int x = (cp.f*point_in_cam(0))/point_in_cam(2) + cp.center_x;
      int y = (cp.f*point_in_cam(1))/point_in_cam(2) + cp.center_y;

      //check if in FOV
      if(x<0 || x >= cp.image_x ||
         y<0 || y >= cp.image_y ) {
        continue;
      }

      double distance = point_in_cam(2)/cp.max_dist;
      //check if distance < what's in the image already
      if(distance < depth_private[x*cp.image_y+y]) {
        depth_private[x*cp.image_y+y] = distance; 
        intensity_private[x*cp.image_y+y] = cloud.points[i].r/255.; 
      }
    }

    //do the reduction by hand here
    #pragma omp critical
    {
      for(int x=0; x<cp.image_x; x++) {
        for(int y=0; y<cp.image_y; y++) {
          if(depth.at<float>(x,y) > depth_private[x*cp.image_y+y]) {
            depth.at<float>(x,y) = depth_private[x*cp.image_y+y];
            intensity.at<float>(x,y) = intensity_private[x*cp.image_y+y];
          }  
        }
      }

    } //end pragma critical

    delete []depth_private; 
    delete []intensity_private; 
    } //end pragma parallel  

}
      
//void CloudRenderer::render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZI> &cloud, Eigen::Affine3d &CameraPose) {
void CloudRenderer::render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                  Eigen::Affine3d &CameraPose) {
    
  depth.create(params_.image_x, params_.image_y, CV_32FC1);
  intensity.create(params_.image_x, params_.image_y, CV_32FC1);
  depth = cv::Scalar(1.0);
  intensity = cv::Scalar(0.0);  
  size_t imsize = params_.image_x*params_.image_y;
#pragma omp parallel shared(depth) shared(intensity) shared(cloud) num_threads(10)
  {

    float *depth_private = new float[imsize];
    for(int q=0; q<imsize; q++) depth_private[q] = 1.0;
    float *intensity_private = new float[imsize];
    for(int q=0; q<imsize; q++) intensity_private[q] = 0.0;

    //std::cerr<< "OpenMP running from process "<<omp_get_thread_num()<<std::endl;
    Eigen::Vector3d point, point_in_cam;

#pragma omp for
    for(int i=0; i<cloud.points.size(); i++) {
      point<<cloud.points[i].x,cloud.points[i].y,cloud.points[i].z;

      //calculate point coordinates relative to camera
      point_in_cam = CameraPose.rotation().transpose()*(point-CameraPose.translation());

      if(point_in_cam(2) <= 0.0) {
        //n_behind++;
        continue; //point behind camera plane
      }

      cv::Point pt;
      this->project(pt,point_in_cam);

      //check if in FOV
      if(pt.x<0 || pt.x >= params_.image_x ||
         pt.y<0 || pt.y >= params_.image_y ) {
        continue;
      }

      double distance = point_in_cam(2)/params_.max_dist;
      //check if distance < what's in the image already
      if(distance < depth_private[pt.x*params_.image_y+pt.y]) {
        depth_private[pt.x*params_.image_y+pt.y] = distance; 
//        intensity_private[pt.x*params_.image_y+pt.y] = cloud.points[i].intensity; 
        intensity_private[pt.x*params_.image_y+pt.y] = cloud.points[i].r/255.; 
      }
    }

    //do the reduction by hand here
#pragma omp critical
    {
      for(int x=0; x<params_.image_x; x++) {
        for(int y=0; y<params_.image_y; y++) {
          if(depth.at<float>(x,y) > depth_private[x*params_.image_y+y]) {
            depth.at<float>(x,y) = depth_private[x*params_.image_y+y];
            intensity.at<float>(x,y) = intensity_private[x*params_.image_y+y];
          }  
        }
      }

    } //end pragma critical

    delete []depth_private; 
    delete []intensity_private; 
  } //end pragma parallel  

}

void CloudRenderer::renderDepth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                                Eigen::Affine3d &CameraPose) {
//void CloudRenderer::renderDepth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZI> &cloud,  Eigen::Affine3d &CameraPose) {

  depth.create(params_.image_x, params_.image_y, CV_32FC1);
  depth = cv::Scalar(1.0);
  size_t imsize = params_.image_x*params_.image_y;
#pragma omp parallel shared(depth) shared(cloud) num_threads(20)
  {

    float *depth_private = new float[imsize];
    for(int q=0; q<imsize; q++) depth_private[q] = 1.0;

    //std::cerr<< "OpenMP running from process "<<omp_get_thread_num()<<std::endl;
    Eigen::Vector3d point, point_in_cam;

#pragma omp for
    for(int i=0; i<cloud.points.size(); i++) {
      point<<cloud.points[i].x,cloud.points[i].y,cloud.points[i].z;

      //calculate point coordinates relative to camera
      point_in_cam = CameraPose.rotation().transpose()*(point-CameraPose.translation());

      if(point_in_cam(2) <= 0.0) {
        //n_behind++;
        continue; //point behind camera plane
      }

      cv::Point pt;
      this->project(pt,point_in_cam);

      //check if in FOV
      if(pt.x<0 || pt.x >= params_.image_x ||
         pt.y<0 || pt.y >= params_.image_y ) {
        continue;
      }

      double distance = point_in_cam(2)/params_.max_dist;
      //check if distance < what's in the image already
      if(distance < depth_private[pt.x*params_.image_y+pt.y]) {
        depth_private[pt.x*params_.image_y+pt.y] = distance; 
      }
    }

    //do the reduction by hand here
#pragma omp critical
    {
      for(int x=0; x<params_.image_x; x++) {
        for(int y=0; y<params_.image_y; y++) {
          if(depth.at<float>(x,y) > depth_private[x*params_.image_y+y]) {
            depth.at<float>(x,y) = depth_private[x*params_.image_y+y];
          }  
        }
      }
    } //end pragma critical
    delete []depth_private; 
  } //end pragma parallel  
}
