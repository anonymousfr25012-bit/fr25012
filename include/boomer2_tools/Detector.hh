#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <boomer2_tools/CloudRenderer.hh>
#include <Eigen/Geometry>

#include <torch/script.h>
#include <torch/torch.h>

namespace graph_matcher {

  //Provides the interface for detecting bolt keypoints
  class DetectorIfce {
    public:
      virtual void detect(cv::Mat &intensity, cv::Mat &depth, std::vector<cv::Point> &keypoints) = 0;
  };

  class FakeDetector : public DetectorIfce {
    public:
      FakeDetector() = delete;
      FakeDetector(FakeDetector &) = delete;
      FakeDetector(std::shared_ptr<graph_matcher::CloudRenderer> &rend, Eigen::Affine3d &pose, pcl::PointCloud<pcl::PointXYZ> &labels): 
        rend_(rend),pose_(pose),labels_(labels) {};

      virtual void detect(cv::Mat &intensity, cv::Mat &depth, std::vector<cv::Point> &keypoints);
    private:
      Eigen::Affine3d pose_;
      std::shared_ptr<graph_matcher::CloudRenderer> rend_;
      pcl::PointCloud<pcl::PointXYZ> labels_;

  };

  class FastrcnnDetector: public DetectorIfce {
    public:
      FastrcnnDetector() = delete;
      FastrcnnDetector(FastrcnnDetector &) = delete;
      FastrcnnDetector(std::shared_ptr<graph_matcher::CloudRenderer> &rend, const std::string model_path, const bool useCPU): 
        rend_(rend) {setModel(model_path, useCPU);};

      virtual void detect(cv::Mat &intensity, cv::Mat &depth, std::vector<cv::Point> &keypoints);      
      int setModel(const std::string model_path, const bool useCPU);
      void getFeatures(Eigen::MatrixXd &features);

    private:
      int getLabels(cv::Mat &intensity, cv::Mat &depth);
      cv::Rect tensorToRect(const at::Tensor& tensor);

      std::shared_ptr<graph_matcher::CloudRenderer> rend_;
      std::vector<cv::Point> labels_;
      
      torch::jit::script::Module module_;
      torch::DeviceType device_;

      Eigen::MatrixXd features_;
  };

}
