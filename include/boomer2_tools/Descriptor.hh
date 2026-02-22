#pragma once

#include <map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <boomer2_tools/CloudRenderer.hh>
#include <boomer2_tools/Features.hh>
#include <pcl/point_types.h>

namespace graph_matcher {

  class DescriptorExtract {
    public:
      /**interface for all descriptors -> calculates the feature pointers at keypoint locations*/
      virtual void extractDescriptors(std::vector<cv::Point> &keypoints, 
                                    std::vector<FeatureBasePtr> &features) = 0;
  };
  typedef std::shared_ptr<DescriptorExtract> DescriptorExtractPtr;

  class EmptyExtract : public DescriptorExtract {
    public:

      EmptyExtract() = delete;
      EmptyExtract(const EmptyExtract&) = delete;
      EmptyExtract(cv::Mat depth, std::shared_ptr<CloudRenderer> render) : 
                   depth_(depth),render_(render) { }

      virtual void extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features);
    private:
      cv::Mat depth_;
      std::shared_ptr<CloudRenderer> render_;

  };

  class OpenCVExtract : public DescriptorExtract {
    public:
      OpenCVExtract() = delete;
      OpenCVExtract(const OpenCVExtract&) = delete;
      OpenCVExtract(cv::Mat depth, cv::Mat intensity, std::string type,
                    std::shared_ptr<CloudRenderer> render) : depth_(depth), 
                    intensity_(intensity), render_(render), type_(type) { }

      virtual void extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features);
    private:
      cv::Mat depth_;
      cv::Mat intensity_;
      std::shared_ptr<CloudRenderer> render_;
      std::string type_;

  };

  class PCLExtract : public DescriptorExtract {
    public:
      PCLExtract() = delete;
      PCLExtract(const PCLExtract&) = delete;
      //PCLExtract(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::string type,
      PCLExtract(cv::Mat &depth, std::string type, std::shared_ptr<CloudRenderer> render) : 
           depth_(depth), render_(render), type_(type) { save_points_ = false; }
      PCLExtract(cv::Mat &depth, std::string type, std::shared_ptr<CloudRenderer> render, bool save_points, std::string save_points_folder) : 
           depth_(depth), render_(render), type_(type), save_points_(save_points), save_points_folder_(save_points_folder){ }
      virtual void extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features);

      void getBolts(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        cloud = cloud_bolts_;
      }
    private:
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bolts_;
      cv::Mat depth_;
      std::shared_ptr<CloudRenderer> render_;
      std::string type_;
      bool save_points_;
      std::string save_points_folder_;
  };

  class NNExtract : public DescriptorExtract {
    public:
      NNExtract() = delete;
      NNExtract(const NNExtract&) = delete;
      
      NNExtract(cv::Mat &depth, std::string type, std::shared_ptr<CloudRenderer> render, Eigen::MatrixXd &features) : 
           depth_(depth), render_(render), type_(type), features_(features) { }

      virtual void extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features);
    private:
      //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
      cv::Mat depth_;
      std::shared_ptr<CloudRenderer> render_;
      std::string type_;
      Eigen::MatrixXd features_;
  };
}
