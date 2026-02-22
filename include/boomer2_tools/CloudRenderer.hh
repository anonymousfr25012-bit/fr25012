#ifndef __CLOUD_RENDERER_HPP
#define __CLOUD_RENDERER_HPP

#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Geometry>

namespace graph_matcher {

  class CamParams {
    public:
      double center_x;
      double center_y;
      double f;
      double max_dist;
      int image_x;
      int image_y;
      bool isFisheye=false;
  }; 

  class CloudRenderer {
    public:
      /** renders a depth and an intensity image from the camera with given parameters 
        * and at a given pose 
        * @Note: static method provided for convenience
        */
      [[deprecated]]
      //static void render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZI> &cloud, CamParams &cp, Eigen::Affine3d &CameraPose);
static void render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZRGB> &cloud, CamParams &cp, Eigen::Affine3d &CameraPose);
      /**************************************************************************/

      //only allow a constructor with camera parameters
      CloudRenderer (CamParams params): params_(params) {}
      CloudRenderer () = delete; 
      CloudRenderer (const CloudRenderer &) = delete; 

      //project a point (in camera coordinates) to a pixel
      inline void project(cv::Point &pixel, Eigen::Vector3d &point) {
        if(params_.isFisheye) {
          double R = point.norm();
          if(R<1e-10) return;
          double phi = atan2(point(1),point(0));
          double theta = acos(point(2)/R);
          double r = params_.f*tan(theta/2);
          pixel.x = r*cos(phi) + params_.center_x;
          pixel.y = r*sin(phi) + params_.center_y;
          //std::cerr<<point.transpose()<<" R= "<<R<<" phi="<<phi<<" theta="<<theta<<" r="<<r<<" px="<<pixel.x<<" py="<<pixel.y<<std::endl;
        } else {
          //standard perspective projection
          pixel.x = (params_.f*point(0))/point(2) + params_.center_x;
          pixel.y = (params_.f*point(1))/point(2) + params_.center_y;
        }
      }

      //back-project a point to a line
      inline bool backproject(cv::Point &pixel, Eigen::Vector3d &line) {
        if(params_.isFisheye) {
          double xprim = pixel.x - params_.center_x;
          double yprim = pixel.y - params_.center_y;
          double sx = xprim > 0 ? 1 : -1;
          //double sy = yprim > 0 ? 1 : -1;
          //double phi = atan2(yprim,xprim);
          double r = sqrt(xprim*xprim+yprim*yprim);
          double cos_theta = cos(2*atan2(params_.f, r));
          double A = sx*sqrt((1/(cos_theta*cos_theta) - 1)/(1+(yprim*yprim)/(xprim*xprim)));
          line<<A, A*yprim/xprim, 1;
        } else {
          //standard perspective projection
          line<<(pixel.x - params_.center_x)/params_.f,
            (pixel.y - params_.center_y)/params_.f,
            1;
          //return true if inside image bounds
        }
        return (pixel.x > 0 && pixel.y > 0 && pixel.x < params_.image_x && pixel.y < params_.image_y);
      }

      /** render using local camera parameters */
      void render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                  Eigen::Affine3d &CameraPose);
//void render(cv::Mat &depth, cv::Mat &intensity, pcl::PointCloud<pcl::PointXYZI> &cloud,                   Eigen::Affine3d &CameraPose);
      /** render just a depth image */
      //void renderDepth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZI> &cloud,                        Eigen::Affine3d &CameraPose);
      void renderDepth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                       Eigen::Affine3d &CameraPose);
      
      inline double getMaxDist() {
        return params_.max_dist;
      }
      CamParams getParams() { return params_; }

    private:
      CamParams params_;

  };
}

#endif
