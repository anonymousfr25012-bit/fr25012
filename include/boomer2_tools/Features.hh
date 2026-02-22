#pragma once

#include <Eigen/Core>
#include <cstdlib>

namespace graph_matcher {

  /** Base class for nodes in the graph */
  class FeatureBase {
    public:
      FeatureBase():type(0) {}
      Eigen::Vector3d pos;
      /** Returns the distance to another feature.
        * 0=same, the more different the features, the higher the distance
        * For the base class, this is always 0
        * To be overloaded by different feature impl
        */
      virtual double compare(const std::shared_ptr<FeatureBase> &other) {
        return 0.0;
        //return ((double)std::rand()/RAND_MAX);
      }
      int id;
      //type uninitialized
      uint8_t type;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  typedef std::shared_ptr<FeatureBase> FeatureBasePtr;

  /** Feature based on the graph connectivity */
  class ConnectivityFeature : public FeatureBase {
    private:
      static constexpr double max_difference = 3.0;
    public:
      ConnectivityFeature() : degree(0) {type=1;}
      ConnectivityFeature(ConnectivityFeature &other) {
        degree=other.degree;
        type=1;
      }
      int degree;
      
      virtual double compare(const FeatureBasePtr &other) {
        std::shared_ptr<ConnectivityFeature> cn = std::dynamic_pointer_cast<ConnectivityFeature> (other);
        if(cn == nullptr) return 1.0;
        double sim=fabsf(degree-cn->degree)/(degree+cn->degree);
        return sim>1 ? 1 : sim;
      }
  };
  typedef std::shared_ptr<ConnectivityFeature> ConnectivityFeaturePtr;
  
  /** Feature based on the graph connectivity */
  class VectorFeature : public FeatureBase {
    public:
      VectorFeature() {type=2;}
      VectorFeature(VectorFeature &other) {
        type=2;
        feature = other.feature;
      }
      Eigen::VectorXd feature;
      virtual double compare(const FeatureBasePtr &other) {
        std::shared_ptr<VectorFeature> cn = std::dynamic_pointer_cast<VectorFeature> (other);
        if(cn == nullptr) return 1.0;
        if(feature.rows()!=cn->feature.rows()) return 1.0;
        return (feature-cn->feature).norm();//(feature.norm()*cn->feature.norm());
      }
  };
  typedef std::shared_ptr<VectorFeature> VectorFeaturePtr;
  
  /** Feature based on the graph connectivity and a Vector feature*/
  class ConnectivityPlusFeature : public VectorFeature {
    private:
      static constexpr double max_difference = 3.0;
    public:
      ConnectivityPlusFeature():degree(0){type=3;}
      ConnectivityPlusFeature(ConnectivityPlusFeature &other) {
        feature = other.feature;
        degree = other.degree;
        type=3;
      }
      int degree;
      virtual double compare(const FeatureBasePtr &other) {
        std::shared_ptr<ConnectivityPlusFeature> cn = std::dynamic_pointer_cast<ConnectivityPlusFeature> (other);
        if(cn == nullptr) return 1.0;
        if(feature.rows()!=cn->feature.rows()) return 1.0;
        
        double sim=fabsf(degree-cn->degree)/(degree+cn->degree);
        return (feature-cn->feature).norm() + sim;
      }
  };
  typedef std::shared_ptr<ConnectivityPlusFeature> ConnectivityPlusFeaturePtr;

}
