#pragma once

#include <map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <boomer2_tools/CloudRenderer.hh>
#include <boomer2_tools/Features.hh>
#include <boomer2_tools/Descriptor.hh>


namespace graph_matcher {

static uint8_t FEATURE_GRAPH_FILE_FORMAT_MAJOR_VERSION=0;
static uint8_t FEATURE_GRAPH_FILE_FORMAT_MINOR_VERSION=1;

  /** Base class for edges */
  class Edge {
    public:
      //the two vertices
      FeatureBasePtr v1, v2;
      //uncertainty associated with the edge
      Eigen::Matrix3d cov_;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  typedef std::shared_ptr<Edge> EdgePtr;

  class FeatureGraph {
    public:
      void addEdge(FeatureBasePtr v1, FeatureBasePtr v2, Eigen::Matrix3d cov);
      void getEdges(FeatureBasePtr v, std::vector<EdgePtr> &edgeList);
      void getVertexMap(std::map<int,FeatureBasePtr> &vmap);
      std::multimap<FeatureBasePtr, EdgePtr>::iterator begin() { return graph_.begin(); }
      std::multimap<FeatureBasePtr, EdgePtr>::iterator end() { return graph_.end(); }
      //sets connectivity information to the features stored in vertices
      void convertToConnectivityFeatures(bool connectivity_plus=false);
      //serialize to a binary file
      bool save(std::string filename);
      //load from a binary file
      bool load(std::string filename);

    private:
      std::multimap<FeatureBasePtr, EdgePtr> graph_;

  };
  typedef std::shared_ptr<FeatureGraph> FeatureGraphPtr;


  /** options for RANSAC */
  struct RANSAC_options {
    ///Euclidean distance to consider points as matched inliers during scoring
    double match_tolerance;
    ///max number of iterations to perform NOTE!! This is the upper limit, in reality we only do 2*possible combinations
    int max_iterations;
    ///random seed
    int seed;
    ///should we perform 1-step ICP refinement and re-scoring based on inliers?
    bool refine_ICP;
    ///Should we only take matches that are both forward and reverse?
    bool forward_reverse;
    ///Should we do simple matching based on threshold?
    bool do_simple_match;
    ///should we instead create a match to the n closest features?
    bool match_n_closest;
    ///If yes, what threshold should we use (range 0 to 1)? pass if dist<thresh
    double dist_threshold;
    ///If instead we are doing unqueness matching, what should we use
    // a = dist closest; b = dist to second closest; a/b < second_closest_thresh
    double second_closest_thresh;
    //how many closest features at most to match?
    int n_closest;

    float random_seed;

    // fix me: change RANSAC_options to another name. we can use RANSAC or TEASER
    bool use_RANSAC;
  };

  class GraphBuilder {

    public:
      GraphBuilder(std::shared_ptr<CloudRenderer> render): render_(render) {};
      GraphBuilder() = delete;
      GraphBuilder(const GraphBuilder &) = delete;

      //builds a feature graph from the keypoints and depth image. Vertices are in camera cordinates
      FeatureGraphPtr buildGraph(cv::Mat &depth, std::vector<cv::Point> &keypoints, DescriptorExtractPtr &extractor);
      //draws the graph onto the provided image. The graph is transformed by T first
      void renderGraph(FeatureGraphPtr &graph, cv::Mat &image, Eigen::Affine3d T=Eigen::Affine3d::Identity(), cv::Scalar color=cv::Scalar(0,0,255), int width=1);

      ///---------------matcher methods-------------------///
      /** Generic matcher based on the descriptor */
      std::vector<std::pair<int,int> > matchGraphs(FeatureGraphPtr &from, FeatureGraphPtr &to, 
                       RANSAC_options &options, Eigen::Affine3d &fromTo);

    private:
      std::shared_ptr<CloudRenderer> render_;

      //Estimates ML transform that minimizs error |T*from - to|
      void getTransform(std::vector<Eigen::Vector3d> &from, 
                        std::vector<Eigen::Vector3d> &to,
                        Eigen::Affine3d &T);

  };

}
