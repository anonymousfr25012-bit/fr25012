#include<boomer2_tools/Descriptor.hh>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>

#include<pcl/features/pfh.h>
#include<pcl/features/fpfh.h>
#include<pcl/features/normal_3d.h>
#include<pcl/features/shot.h>

#include <sstream>      // std::stringstream
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <Eigen/Eigen>

using namespace graph_matcher;

void EmptyExtract::extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features) {

  for(auto itr=keypoints.begin(); itr!=keypoints.end(); ++itr) {
    //project keypoint to 3D
    Eigen::Vector3d v1;
    render_->backproject(*itr,v1);
    v1 = v1*depth_.at<float>(itr->x,itr->y)*render_->getMaxDist(); //scale ray by depth

    FeatureBasePtr f1 (new FeatureBase());
    f1->pos=v1; 
    f1->id = itr-keypoints.begin();

    features.push_back(f1);
  }

}
      

void OpenCVExtract::extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features) {

  std::vector<cv::KeyPoint> cv_keypoints;

  int patch_size = 10;
  for(auto itr=keypoints.begin(); itr!=keypoints.end(); itr++) {
    double depth = depth_.at<float>(itr->x,itr->y)*render_->getMaxDist(); 
    cv::KeyPoint kp (*itr, patch_size*depth);
    cv_keypoints.push_back(kp);
  }

  //ORB features
  /*
  cv::Ptr<cv::FeatureDetector> featureDetector = cv::ORB::create(500, 1.2, 8, patch_size, 0, 
                                                        2, cv::ORB::HARRIS_SCORE,patch_size, 20);
  
  featureDetector->detect(intensity_, cv_keypoints);
  std::cerr<<"Detected "<<cv_keypoints.size()<<" keypoints\n";

  cv::Ptr<cv::DescriptorExtractor> featureExtractor = cv::ORB::create(500, 1.2, 8, patch_size, 0, 
                                                        2, cv::ORB::HARRIS_SCORE,patch_size, 20);
  */
  cv::Ptr<cv::DescriptorExtractor> featureExtractor;
  if(type_ == "ORB") { 
    featureExtractor = cv::ORB::create(500, 1.2, 8, patch_size, 0, 
        2, cv::ORB::HARRIS_SCORE,patch_size, 20);
  } else if(type_ == "SIFT") {
    featureExtractor = cv::SIFT::create();
  } else {
    //default to BRISK
    featureExtractor = cv::BRISK::create();
  }

  cv::Mat descriptors;
  featureExtractor->compute(intensity_, cv_keypoints, descriptors);

  keypoints.clear();

  //std::cerr<<"Desriptors are "<<descriptors.rows<<", "<<descriptors.cols<<std::endl;
  for(int it=0; it<cv_keypoints.size(); ++it) {
    //project keypoint to 3D
    Eigen::Vector3d v1;
    cv::Point kpt = cv_keypoints[it].pt;
    render_->backproject(kpt,v1);
    v1 = v1*depth_.at<float>(kpt.x,kpt.y)*render_->getMaxDist(); 
   
    VectorFeaturePtr f1 (new VectorFeature());
    f1->pos=v1; 
    f1->id = it;
    cv::cv2eigen(descriptors.row(it).t(), f1->feature);

    //FIXME for BRISK
    f1->feature = f1->feature/255.;

    cv::KeyPoint kp = cv_keypoints[it];
    std::cerr<<"kp "<<it<<" = "<<kp.pt.x<<" "<<kp.pt.y<<" "<<kp.size<<" "<<kp.angle<<"\n";
    //std::cerr<<"kp "<<it<<" = "<<f1->feature.transpose()<<std::endl;

    if(f1->feature.norm() < 0.01) { continue; }
    FeatureBasePtr f1_base (f1);
    features.push_back(f1_base);
    keypoints.push_back(kpt);
  }
  std::cerr<<keypoints.size()<<" keypoints with descriptors extracted\n"; 
  //std::cerr<<"----------------------------------------------------------------------\n\n"; 
}



void PCLExtract::extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features) {

  std::vector<cv::Point> keypoints_new;

  float normals_radius = 0.20;
  int pixel_n = 40;
  // float normals_radius = 0.15;
  // int pixel_n = 60;
 
  double volume_factor = 1./(8*normals_radius*normals_radius*normals_radius); 
  size_t point_no = 0;
  //make some pcl keypoints
  // fix me: save all detected bolts
  if (save_points_ && save_points_folder_ != std::string("") &&!boost::filesystem::exists(save_points_folder_)) {
    boost::filesystem::create_directories(save_points_folder_);
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bolts(new pcl::PointCloud<pcl::PointXYZ>());

  #pragma omp parallel for
  for(auto itr=keypoints.begin(); itr!=keypoints.end(); ++itr) {
    //project keypoint to 3D
    Eigen::Vector3d v1;
    render_->backproject(*itr,v1);
    v1 = v1*depth_.at<float>(itr->x,itr->y)*render_->getMaxDist(); 

    // if(std::isnan(v1.maxCoeff())||std::isinf(v1.maxCoeff())) {
    if(v1.array().isInf().any() || v1.array().isNaN().any()) {
      std::cerr<<"Keypoint pixel at "<<itr->x<<" "<<itr->y<<" depth "<<depth_.at<float>(itr->x,itr->y)*render_->getMaxDist()<<" projects to "<<v1.transpose()<<" skipping\n";
      continue;
    }

    pcl::PointXYZ pt;
    pt.x = v1(0);
    pt.y = v1(1);
    pt.z = v1(2);

    //render ourselves a local point cloud
    int idxPt=-1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>() );
    cloud->push_back(pt); 

    #pragma omp parallel for
    for(int i=-pixel_n/2; i<pixel_n/2; i++) {
      for(int j=-pixel_n/2; j<pixel_n/2; j++) {
        Eigen::Vector3d vi;
        cv::Point pt_2d (itr->x+i, itr->y+j);
        if(render_->backproject(pt_2d,vi)) {
          float depth = depth_.at<float>(pt_2d.x,pt_2d.y);
          if(depth<1.0f && depth>0.0f) {
            vi = vi*depth*render_->getMaxDist();
            if(std::isnan(vi(0))||std::isinf(vi(0))) continue;
            if(std::isnan(vi(1))||std::isinf(vi(1))) continue;
            if(std::isnan(vi(2))||std::isinf(vi(2))) continue;
            pcl::PointXYZ pi;
            pi.x = vi(0);
            pi.y = vi(1);
            pi.z = vi(2);
            // exclude points which are far from center.
            auto diff = pi.getVector3fMap() - pt.getVector3fMap();

            if (diff.norm() > 5) {
                continue;
            }
            cloud->push_back(pi); 
            
          } 
          if(i==0 && j==0) {
            idxPt = cloud->points.size()-1;
          }
        }
      }
    }
    //std::cerr<<"got local cloud of size "<<cloud->size()<<std::endl;
    
    // *cloud_bolts = *cloud_bolts + *cloud;
    // fix me: save all detected bolts
    // if (save_points_) {
    //   std::string filename = "bolt_" + std::to_string(point_no++) + ".pcd";
    //   pcl::io::savePCDFileASCII(save_points_folder_+"/"+filename, *cloud);
    // }    

    // Create the normal estimation class, and pass the input dataset to it
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
    std::vector<float>pointRadiusSquaredDistance;

    //find neighbors
    // std::cerr << "find neighbors..." << std::endl;
    // std::cerr << pt << std::endl;
    
    if ( tree->radiusSearch (pt, normals_radius*2, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
    {

      // std::cerr << "find neighbors!" << std::endl;
      for(auto pitr=pointIdxRadiusSearch.begin(); pitr<pointIdxRadiusSearch.end(); ++pitr) {
        Eigen::Vector3d vi;
        auto point = (*cloud)[*pitr];
        vi<<point.x,point.y,point.z;
        cv::Point back;
        render_->project(back,vi);
        depth_.at<float>(back.x,back.y)=1.0;
      }
      std::stringstream ss;
      std::string fname;
      ss<<"points_"<<itr-keypoints.begin()<<".ply";
      ss>>fname;
     
      pcl::PointCloud<pcl::PointNormal> cloud_with_normals;
      for(auto pitr=pointIdxRadiusSearch.begin(); pitr<pointIdxRadiusSearch.end(); ++pitr) {
        pcl::PointNormal pn;
        auto point = (*cloud)[*pitr];
        pn.x = point.x;
        pn.y = point.y;
        pn.z = point.z;
        pcl::Normal n = cloud_normals->points[*pitr];
        pn.normal_x = n.normal_x;
        pn.normal_y = n.normal_y;
        pn.normal_z = n.normal_z;
        pn.curvature = n.curvature;
        cloud_with_normals.push_back(pn);
      }
       
      // pcl::PCDWriter writer;
      // writer.write (fname, cloud_with_normals);
      // pcl::io::savePLYFileASCII(fname, cloud_with_normals);
      // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Visualization"));
      // viewer->setBackgroundColor(0, 0, 0);
      // viewer->addPointCloud<pcl::PointXYZ>(cloud_with_normals, "cloud_with_normals");
      

      VectorFeaturePtr f1 (new VectorFeature());
      f1->pos = v1; 
      f1->id = itr-keypoints.begin();
      
      if(type_ == "PFH") {

        pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
        Eigen::VectorXf pfh_histogram(125);
        pfh.setRadiusSearch(2*normals_radius); //0.15
        pfh.computePointPFHSignature(*cloud,*cloud_normals, pointIdxRadiusSearch, 5, pfh_histogram);
        
        // std::cerr<<"For point "<<v1.transpose()<<" neighbors are "<<pointIdxRadiusSearch.size()<<std::endl;
        // std::cerr<<"Histogram "<<pfh_histogram.transpose()<<std::endl;
        //add feature to result
        f1->feature = pfh_histogram.cast<double>();
      } else if (type_ == "FPFH") {  
        // throw std::runtime_error("Not implemented yet");
        // std::cerr << "using FPFH feature" << std::endl;
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(cloud_normals);
        // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

        // Create an empty kdtree representation, and pass it to the FPFH estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

        fpfh.setSearchMethod(tree);

        // Output datasets
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

        // Use all neighbors in a sphere of radius 5cm
        // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
        fpfh.setRadiusSearch(2*normals_radius); //0.15

        // Compute the features
        fpfh.compute(*fpfhs);
        // std::cerr << "Cloud size is " << cloud->size() << " FPFH output points.size (): " << fpfhs->points.size() << std::endl;
        Eigen::VectorXf fpfh_histogram(33);
        
        for (int i = 0; i < 33; ++i) {
          fpfh_histogram[i] = fpfhs->points[0].histogram[i];
        }
        // Save the FPFH feature to f1->feature
        f1->feature = fpfh_histogram.cast<double>();
      } else if (type_ == "SHOT") {
        pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimation; 
        shotEstimation.setInputCloud(cloud);
        shotEstimation.setInputNormals(cloud_normals);
        shotEstimation.setSearchMethod(tree);
        
        pcl::PointCloud<pcl::SHOT352>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT352>);
        shotEstimation.setKSearch(0);
        shotEstimation.setRadiusSearch(2*normals_radius);

        // Actually compute the spin images
        shotEstimation.compute (*shotFeatures);
        //std::cout << "Cloud size is "<<cloud->size()<<" SHOT output points.size (): " << shotFeatures->points.size () << std::endl;

        // Display and retrieve the SHOT descriptor for the first point.
        pcl::SHOT352 descriptor = shotFeatures->points[idxPt];
        //std::cout << descriptor << std::endl;

        //shotEstimation.computePointSHOT(idxPt,pointIdxRadiusSearch,pointRadiusSquaredDistance,shot);
        //shot = descriptor;
        Eigen::Map<Eigen::VectorXf> shot(descriptor.descriptor, 352);
        f1->feature = shot.cast<double>();
        //std::cerr<<"For point "<<v1.transpose()<<" neighbors are "<<pointIdxRadiusSearch.size()<<std::endl;
        //std::cerr<<"Histogram "<<shot.transpose()<<std::endl;
      } else if (type_ == "NDT") {
        //simple local gaussian feature
        if(pointIdxRadiusSearch.size() < 6) continue;

        //get ML Gaussian:
        //compute mean
        Eigen::Vector3d mean,tmp,mean_normal;  
        double mean_curve; 
        mean<<0,0,0;   
        for(auto pitr=pointIdxRadiusSearch.begin(); pitr<pointIdxRadiusSearch.end(); ++pitr) {
          auto point = (*cloud)[*pitr];
          tmp<<point.x,point.y,point.z;
          pcl::Normal n = cloud_normals->points[*pitr];
          mean_curve += n.curvature;
          tmp<<n.normal_x,n.normal_y,n.normal_z;
          mean_normal += tmp;
        }
        mean /= pointIdxRadiusSearch.size();
        mean_curve /= pointIdxRadiusSearch.size();
        mean_normal /= pointIdxRadiusSearch.size();
        //compute Covariance
        Eigen::MatrixXd mp;

        mp.resize(pointIdxRadiusSearch.size(),3);
        for(unsigned int i=0; i< pointIdxRadiusSearch.size(); i++)
        {
          auto point = (*cloud)[pointIdxRadiusSearch[i]];
          mp(i,0) = point.x - mean(0);
          mp(i,1) = point.y - mean(1);
          mp(i,2) = point.z - mean(2);
        }
        Eigen::Matrix3d cov_ = mp.transpose()*mp/(pointIdxRadiusSearch.size()-1);

        //compute Eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Sol (cov_);

        Eigen::Matrix3d evecs;
        Eigen::Vector3d evals;

        evecs = Sol.eigenvectors().real();
        evals = Sol.eigenvalues().real();

        int idMax,idMin;
        double minEval = evals.minCoeff(&idMin);
        double maxEval = evals.maxCoeff(&idMax);

        //fill in feature:
        f1->feature = Eigen::VectorXd(4);
        //volume of ellipsoid
        f1->feature(0) = evals(0)*evals(1)*evals(2)*volume_factor; //maxEval;
        //f1->feature(1) = maxEval;
        //normal dot with down gravity
        f1->feature(1) = fabsf(evecs.col(idMin).dot(Eigen::Vector3d::UnitZ()));
        //mean curvature
        f1->feature(2) = 10*mean_curve;
        f1->feature(3) = fabsf(mean_normal.dot(Eigen::Vector3d::UnitZ()));
        //std::cerr<<f1->feature.transpose()<<std::endl;
      }

      if(f1->feature.norm() < 1e-2) continue; //ignore descriptors that are too tiny
      FeatureBasePtr f1_base (f1);
      features.push_back(f1_base);
      keypoints_new.push_back(*itr);
    }
// pcl::io::savePCDFile("cloud_bolts.pcd", *cloud_bolts);
  // cloud_bolts_ = cloud_bolts;
  }
  
  keypoints=keypoints_new;
}

void NNExtract::extractDescriptors(std::vector<cv::Point> &keypoints, 
          std::vector<FeatureBasePtr> &features) {
    
    std::vector<cv::KeyPoint> cv_keypoints;

    int patch_size = 10;
    for(auto itr=keypoints.begin(); itr!=keypoints.end(); itr++) {
      double depth = depth_.at<float>(itr->x,itr->y)*render_->getMaxDist(); 
      cv::KeyPoint kp (*itr, patch_size*depth);
      cv_keypoints.push_back(kp);
    }

    keypoints.clear();
    for(int it=0; it<cv_keypoints.size(); ++it) {
      //project keypoint to 3D
      Eigen::Vector3d v1;
      cv::Point kpt = cv_keypoints[it].pt;
      render_->backproject(kpt,v1);
      v1 = v1*depth_.at<float>(kpt.x,kpt.y)*render_->getMaxDist(); 
    
      VectorFeaturePtr f1 (new VectorFeature());
      f1->pos=v1; 
      f1->id = it;

      //FIXME for BRISK
      f1->feature = features_.row(it);
      std::cerr << features_.row(it).size() << std::endl;

      cv::KeyPoint kp = cv_keypoints[it];
      std::cerr<<"kp "<<it<<" = "<<kp.pt.x<<" "<<kp.pt.y<<" "<<kp.size<<" "<<kp.angle<<"\n";
      //std::cerr<<"kp "<<it<<" = "<<f1->feature.transpose()<<std::endl;
      
      FeatureBasePtr f1_base (f1);
      features.push_back(f1_base);
      keypoints.push_back(kpt);
    }
  std::cerr<<keypoints.size()<<" keypoints with descriptors extracted\n"; 
}
