#include<boomer2_tools/GraphBuilder.hh>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <set>
#include <Eigen/SVD>

#include<boomer2_tools/Descriptor.hh>

#include <omp.h>

using namespace graph_matcher;

void FeatureGraph::addEdge(FeatureBasePtr v1, FeatureBasePtr v2, Eigen::Matrix3d cov) {

      EdgePtr e (new Edge());
      e->v1 = v1;
      e->v2 = v2;
      e->cov_ = cov;
      
      graph_.insert(std::pair{v1,e});
      //TODO: insert reverse edge as well?
}

void FeatureGraph::getEdges(FeatureBasePtr v, std::vector<EdgePtr> &edgeList) {

}
      
void FeatureGraph::getVertexMap(std::map<int,FeatureBasePtr> &vmap) {
  vmap.clear();
  for(auto itr=graph_.begin(); itr!=graph_.end(); itr++) {
    vmap[itr->second->v1->id] = itr->second->v1;
    vmap[itr->second->v2->id] = itr->second->v2;
  }
}

void FeatureGraph::convertToConnectivityFeatures(bool connectivity_plus) {
  //iterate through edges
  std::multimap<FeatureBasePtr, EdgePtr> new_graph_;
  for(auto itr=graph_.begin(); itr!=graph_.end(); itr++) {
    ConnectivityFeaturePtr cn_first (new ConnectivityFeature);
    cn_first->id = (itr)->second->v1->id;
    cn_first->pos = (itr)->second->v1->pos;
    //count how many vertices we connect to
    for(auto jtr=graph_.begin() ; jtr!=graph_.end(); jtr++) {
      if(jtr->second->v1->id == cn_first->id || jtr->second->v2->id == cn_first->id) {
        cn_first->degree++;
        //std::cout<<"also in "<<jtr->second->v1->id<<" - "<<jtr->second->v2->id<<std::endl;
      }
    }

    ConnectivityFeaturePtr cn_second (new ConnectivityFeature);
    cn_second->id = (itr)->second->v2->id;
    cn_second->pos = (itr)->second->v2->pos;
    //count how many vertices we connect to
    for(auto jtr=graph_.begin() ; jtr!=graph_.end(); jtr++) {
      if(jtr->second->v1->id == cn_second->id || jtr->second->v2->id == cn_second->id) {
        cn_second->degree++;
        //std::cout<<"also in "<<jtr->second->v1->id<<" - "<<jtr->second->v2->id<<std::endl;
      }
    }

    if(!connectivity_plus) {
      EdgePtr e (new Edge());
      e->v1 = FeatureBasePtr(cn_first);
      e->v2 = FeatureBasePtr(cn_second);
      e->cov_ = itr->second->cov_;
      
      new_graph_.insert(std::pair{FeatureBasePtr(cn_first),e});
    } else {

      ConnectivityPlusFeaturePtr cn_plus_first (new ConnectivityPlusFeature);
      ConnectivityPlusFeaturePtr cn_plus_second (new ConnectivityPlusFeature);
      cn_plus_first->id = cn_first->id;
      cn_plus_first->pos = cn_first->pos;
      cn_plus_first->degree = cn_first->degree;
      cn_plus_second->id = cn_second->id;
      cn_plus_second->pos = cn_second->pos;
      cn_plus_second->degree = cn_second->degree;
     
      //cast itr featureBase to Vector 
      VectorFeaturePtr vpt_first = std::dynamic_pointer_cast<VectorFeature> (itr->second->v1);
      VectorFeaturePtr vpt_second = std::dynamic_pointer_cast<VectorFeature> (itr->second->v2);

      if(vpt_first == nullptr || vpt_second == nullptr) continue;
      cn_plus_first->feature = vpt_first->feature;
      cn_plus_second->feature = vpt_second->feature;

      EdgePtr e (new Edge());
      e->v1 = FeatureBasePtr(cn_plus_first);
      e->v2 = FeatureBasePtr(cn_plus_second);
      e->cov_ = itr->second->cov_;
      new_graph_.insert(std::pair{FeatureBasePtr(cn_plus_first),e});
    }

  }
  graph_=new_graph_;
}
      
//serialize to a binary file
bool FeatureGraph::save(std::string filename){

  std::ofstream out(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

  std::map<int,FeatureBasePtr> vmap;
  //first we collect all vertices in a list
  for(auto itr=graph_.begin(); itr!=graph_.end(); itr++) {
    vmap[itr->second->v1->id] = itr->second->v1;
    vmap[itr->second->v2->id] = itr->second->v2;
  }

  //write file format version
  out.write((char*)&FEATURE_GRAPH_FILE_FORMAT_MAJOR_VERSION, sizeof(uint8_t));
  out.write((char*)&FEATURE_GRAPH_FILE_FORMAT_MINOR_VERSION, sizeof(uint8_t));

  int n_vert = vmap.size();
  int n_edge = graph_.size();

  std::cerr<<"Saving graph with "<<n_vert<<" vertices and "<<n_edge<<" edges\n";
  //write number of vertices
  out.write((char*)&n_vert, sizeof(n_vert));
  //write number of edges
  out.write((char*)&n_edge, sizeof(n_edge));

  //write vertices
  for(auto jtr=vmap.begin(); jtr!=vmap.end(); jtr++) {
    //vertex ID, vertex feature TYPE
    out.write((char*)&jtr->first, sizeof(jtr->first));
    out.write((char*)&jtr->second->type, sizeof(jtr->second->type));
    //vertex 3D position
    out.write((char*)jtr->second->pos.data(), 3*sizeof(double));
    if(jtr->second->type == 0) {
      std::cerr<<"Trying to save base feature might not be a good idea\n";
    } 
    else if(jtr->second->type == 1) {
      //connectivity feature
      ConnectivityFeaturePtr cn = std::dynamic_pointer_cast<ConnectivityFeature>(jtr->second);
      if(cn!=nullptr) {
        //vertex DEGREE
        out.write((char*)&cn->degree, sizeof(cn->degree));
      } else {
        std::cerr<<"Inconsistent feature type and feature pointer!\n";
        return false;
      }
    } 
    else if(jtr->second->type == 2) {
      //vector feature
      VectorFeaturePtr fn = std::dynamic_pointer_cast<VectorFeature>(jtr->second);
      if(fn!=nullptr) {
        //vertex FEATURE SIZE, vertex FEATURE
        int feature_size = fn->feature.rows();
        out.write((char*)&feature_size, sizeof(feature_size));
        out.write((char*)fn->feature.data(), sizeof(double)*fn->feature.rows());
      } else {
        std::cerr<<"Inconsistent feature type and feature pointer!\n";
        return false;
      }
    } 
    else if(jtr->second->type == 3) {
      //vector+connectivity feature
      ConnectivityPlusFeaturePtr cfn = std::dynamic_pointer_cast<ConnectivityPlusFeature>(jtr->second);
      if(cfn!=nullptr) {
        //vertex DEGREE
        out.write((char*)&cfn->degree, sizeof(cfn->degree));
        //vertex FEATURE SIZE, vertex FEATURE
        int feature_size = cfn->feature.rows();
        out.write((char*)&feature_size, sizeof(feature_size));
        out.write((char*)cfn->feature.data(), sizeof(double)*cfn->feature.rows());
      
      } else {
        std::cerr<<"Inconsistent feature type and feature pointer!\n";
        return false;
      }
    } else
    {
      return false;
    }
  }

  //write edges
  for(auto itr=graph_.begin(); itr!=graph_.end(); itr++) {
    //edge two IDs
    out.write((char*)&itr->second->v1->id, sizeof(itr->second->v1->id));
    out.write((char*)&itr->second->v2->id, sizeof(itr->second->v2->id));
    //edge covariance matrix, 3x3
    out.write((char*)(itr->second->cov_.data()), 9*sizeof(double));
  }

  out.close();
  return true;
}

//load from a binary file
bool FeatureGraph::load(std::string filename) {
  //clear all current data
  graph_.clear();

  std::ifstream in(filename, std::ofstream::in | std::ofstream::binary);

  if(!in.is_open() || in.eof()) return false;

  uint8_t major_version=255, minor_version=255;
  //read file format version
  in.read((char*)&major_version, sizeof(uint8_t));
  in.read((char*)&minor_version, sizeof(uint8_t));

  if(major_version!=FEATURE_GRAPH_FILE_FORMAT_MAJOR_VERSION) {
    std::cerr<<"Cannot read incompatible version of file. File version "<<major_version
             <<"."<<minor_version<<" our version "<< FEATURE_GRAPH_FILE_FORMAT_MAJOR_VERSION
             <<"."<<FEATURE_GRAPH_FILE_FORMAT_MINOR_VERSION<<std::endl;
    return false;
  }

  int n_vert, n_edge;
  in.read((char*)&n_vert, sizeof(n_vert));
  in.read((char*)&n_edge, sizeof(n_edge));
  std::cerr<<"Reading "<<n_vert<<" vertices and "<<n_edge<<" edges\n";

  std::map<int,FeatureBasePtr> vmap;
  uint8_t type;
  int id, degree, feature_size;
  Eigen::Vector3d pos;

  for(int i=0; i<n_vert; i++) {

    //vertex ID, vertex feature TYPE
    in.read((char*)&id, sizeof(id));
    in.read((char*)&type, sizeof(type));
    //ivertex position
    in.read((char*)pos.data(), 3*sizeof(double));

    //Connectivity Feature
    if(type==1) {
      in.read((char*)&degree, sizeof(degree));
      
      auto cn = std::make_shared<ConnectivityFeature>();
      cn->pos = pos;
      cn->id = id;
      cn->type = type;
      cn->degree = degree;

      vmap[id] = cn;
    }
    //Vector Feature
    if(type==2) {
      auto cf = std::make_shared<VectorFeature>();
      
      in.read((char*)&feature_size, sizeof(feature_size));
      
      cf->feature = Eigen::VectorXd(feature_size);
      in.read((char*)cf->feature.data(), feature_size*sizeof(double));
      
      cf->pos = pos;
      cf->id = id;
      cf->type = type;
      
      vmap[id] = cf;
    }
    //Connectivity Vector Feature
    if(type==3) {
      auto cfn = std::make_shared<ConnectivityPlusFeature>();
      
      in.read((char*)&degree, sizeof(degree));
      in.read((char*)&feature_size, sizeof(feature_size));
      
      cfn->feature = Eigen::VectorXd(feature_size);
      in.read((char*)cfn->feature.data(), feature_size*sizeof(double));
      
      cfn->pos = pos;
      cfn->id = id;
      cfn->type = type;
      cfn->degree = degree;
      
      vmap[id] = cfn;
    }
    if(type == 0) {
      std::cerr<<"Base feature not loadable\n";
    }
    if(type>3) {
      std::cerr<<"Feature type unknown\n";
    }
  }

  Eigen::Matrix3d cov;
  int id1,id2;
  //edges
  for(int i=0; i<n_edge; i++) {
    //edge two IDs
    in.read((char*)&id1, sizeof(id1));
    in.read((char*)&id2, sizeof(id2));
    in.read((char*)cov.data(),9*sizeof(double));

    this->addEdge(vmap[id1],vmap[id2],cov);
  }

  return true;
}

//Graph is built with vertices in camera coordinates
FeatureGraphPtr GraphBuilder::buildGraph(cv::Mat &depth, std::vector<cv::Point> &keypoints, DescriptorExtractPtr &extractor) {

  double MAX_DIST = 3.0; //maximum 3 meters between features
  double eps = 0.1;      //maximum 10cm deviation from line
  double MAX_OUTLIER_RATIO = 0.4; //maximum portion of outliers
  Eigen::Matrix3d cov = 0.1*Eigen::Matrix3d::Identity();
  FeatureGraphPtr graph (new FeatureGraph());

  std::vector<FeatureBasePtr> descriptors;
  extractor->extractDescriptors(keypoints,descriptors); 

  auto descr_itr = descriptors.begin();
  auto descr_jtr = descriptors.begin();

  int vertex_id=0;

  for(auto itr=keypoints.begin(); itr!=keypoints.end(); ++itr,++descr_itr) {
    descr_jtr = descr_itr+1;

    //check if we can draw an edge for all following points
    for(auto jtr=itr+1; jtr!=keypoints.end(); ++jtr,++descr_jtr) {
      if(((*descr_itr)->pos-(*descr_jtr)->pos).norm() > MAX_DIST) continue;

      //outlier, ignore if just using FPFH
      Eigen::Vector3d dir = ((*descr_jtr)->pos-(*descr_itr)->pos);
      dir.normalize();

      //for drawing points are in y,x order for some reason
      cv::Point start, end;
      start.x = itr->y; start.y = itr->x;
      end.x = jtr->y; end.y = jtr->x;
      //create an iterator for the line from itr to jtr
      auto line = cv::LineIterator(depth, start, end);
      int n_outlier=0, n_overall=0;
      //std::cerr<<"Start "<<itr->x<<" "<<itr->y<<" --> ";
      while (line.pos() != end) { //there doesn't seem to be a better way to find the end
        Eigen::Vector3d vx;
        cv::Point pos; 
        //here as well we need to flip back
        pos.x = line.pos().y; pos.y = line.pos().x;
        //std::cerr<<"("<<pos.x<<","<<pos.y<<")";
        render_->backproject(pos,vx);
        vx = vx*depth.at<float>(pos.x, pos.y)*render_->getMaxDist();
        
        //formula for distance
        double distance = ((vx-(*descr_itr)->pos)-((vx-(*descr_itr)->pos).dot(dir))*dir).norm();
        if(distance>eps) {
          n_outlier++;
        }
        n_overall++;
        line++;
      }

      double outlier_ratio = (double)n_outlier/(double)(n_overall+1);
      //std::cerr<<"Outlier ratio is "<<outlier_ratio<<std::endl;
      if(outlier_ratio > MAX_OUTLIER_RATIO) continue;
      
      //add edges to the graph
      graph->addEdge(*descr_itr,*descr_jtr,cov);
    }
  }

  return graph;
}

void GraphBuilder::renderGraph(FeatureGraphPtr &graph, cv::Mat &image, Eigen::Affine3d T, cv::Scalar color, int width) {

  for(auto itr=graph->begin(); itr!=graph->end(); itr++) {
    Eigen::Vector3d v1 = itr->second->v1->pos;
    Eigen::Vector3d v2 = itr->second->v2->pos;

    v1 = T*v1;
    v2 = T*v2;

    cv::Point p1,p2;
    render_->project(p1,v1);
    render_->project(p2,v2);

    //clip p1,p2 within image size
    cv::Point p1_draw,p2_draw;
    p1_draw.x = MIN(image.cols, MAX(p1.y, 0)); 
    p1_draw.y = MIN(image.rows, MAX(p1.x, 0)); 
    p2_draw.x = MIN(image.cols, MAX(p2.y, 0)); 
    p2_draw.y = MIN(image.rows, MAX(p2.x, 0));

    cv::line(image, p1_draw, p2_draw, color, width);
  }
}
      
void GraphBuilder::getTransform(std::vector<Eigen::Vector3d> &from, 
                        std::vector<Eigen::Vector3d> &to,
                        Eigen::Affine3d &T) {

    if(from.size()!=to.size()) return;

    const int N = from.size();
    Eigen::MatrixXd p1(3,N), p2(3,N);

    Eigen::Matrix3d PT;
    Eigen::Vector3d v1, v2, m1, m2;
    m1<<0,0,0;
    m2<<0,0,0;
   
    for(int q=0; q<N; q++) {
      m1 = m1+from[q];
      m2 = m2+to[q];
      p1.col(q) = from[q];
      p2.col(q) = to[q];
    }
    //std::cerr<<"p1 = \n"<<p1<<"\n";
    //std::cerr<<"p2 = \n"<<p2<<"\n";
    m1=m1/3;
    m2=m2/3;
    //std::cerr<<"m1 = "<<m1.transpose()<<"\n";
    //std::cerr<<"m2 = "<<m2.transpose()<<"\n";
    p1 = p1.colwise() - m1;
    p2 = p2.colwise() - m2;
    //std::cerr<<"p1c = \n"<<p1<<"\n";
    //std::cerr<<"p2c = \n"<<p2<<"\n";

    PT=p1 * p2.transpose();
    //std::cerr<<"pT = \n"<<PT<<"\n";
    // Compute the Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3> > svd(
        PT, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3> u = svd.matrixU();
    Eigen::Matrix<double, 3, 3> v = svd.matrixV();

    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0) {
      for (int x = 0; x < 3; ++x)
        v(x, 2) *= -1;
    }

    Eigen::Matrix<double, 3, 3> R = v * u.transpose();

    // Return the correct transformation
    T.matrix().topLeftCorner(3, 3) = R;
    const Eigen::Matrix<double, 3, 1> Rc(R * m1);
    T.matrix().block(0, 3, 3, 1) = m2 - Rc;

}


/**---------------------Generic matcher---------------------------*/
// fix me: add flag to choose teaser or ransac
std::vector<std::pair<int,int> > GraphBuilder::matchGraphs(FeatureGraphPtr &from, FeatureGraphPtr &to, RANSAC_options &options, Eigen::Affine3d &fromTo)
{
  std::set<int> forward_vertices, reverse_vertices;
  std::vector<std::pair<int,int> > matches;
  std::map<int, FeatureBasePtr> from_features, to_features;
  from->getVertexMap(from_features);
  to->getVertexMap(to_features);

  //forward matches will be same as reverse for now
  std::vector<std::pair<FeatureBasePtr, FeatureBasePtr> > forward, reverse;

  FeatureBasePtr match;
  typedef std::pair<double,FeatureBasePtr> RankFeature;

  
  for (auto itr=from_features.begin(); itr!=from_features.end(); itr++) {
    double top_score=INT_MAX, second_score=INT_MAX;
    //std::cerr<<"Vertex "<<itr->first<<": ";
    std::vector<RankFeature> closest_list;
  
    for (auto jtr=to_features.begin(); jtr!=to_features.end(); jtr++) {
      double dist = itr->second->compare(jtr->second);
      if(options.do_simple_match && dist<options.dist_threshold) {
        FeatureBasePtr vertex = itr->second;
        match = jtr->second;
        //std::cerr<<"Match to "<<match->pos.transpose()<<" = "<<dist<<" ";
        forward.push_back(std::pair<FeatureBasePtr, FeatureBasePtr>(vertex, match));
        forward_vertices.insert(vertex->id);
        reverse_vertices.insert(match->id);
      } 
      if(!options.do_simple_match)
      {
        if(options.match_n_closest) {
          //we need to maintain a list and sort it
          closest_list.push_back(RankFeature(dist,jtr->second));
        } else {
          //keep track of only best and second best match
          if(dist<top_score) {
            second_score = top_score;
            top_score = dist;
            match = jtr->second;
          } else if(dist<second_score) {
            second_score = dist;
          }
        }
      }
    }
    //if we are doing second best matching now's the time to check
    if(!options.do_simple_match) {
      if(options.match_n_closest) {
        //sort list
        std::sort(closest_list.begin(), closest_list.end(), [](RankFeature a, RankFeature b) 
            { return a.first < b.first; });
        for(int m=0; m<options.n_closest && m<closest_list.size(); m++) {
          //std::cerr<<"top "<<closest_list[0].first<<" current "<<closest_list[m].first<<std::endl;
          
          FeatureBasePtr vertex = itr->second;
          forward.push_back(std::pair<FeatureBasePtr, FeatureBasePtr>(vertex, closest_list[m].second));
          forward_vertices.insert(vertex->id);
          reverse_vertices.insert(closest_list[m].second->id);
        }
      } else {
        if((top_score+1e-10)/(second_score+1e-10) < options.second_closest_thresh) {
          //std::cerr<<"top "<<top_score<<" second "<<second_score<<" ratio "<<(top_score+1e-10)/(second_score+1e-10)<<" match "<<itr->second->id<<" to "<<match->id<<std::endl;
          //we have a winner
          FeatureBasePtr vertex = itr->second;
          forward.push_back(std::pair<FeatureBasePtr, FeatureBasePtr>(vertex, match));
          forward_vertices.insert(vertex->id);
          reverse_vertices.insert(match->id);
        }
      }
    }
  }

  std::cerr<<"----------------Reverse-----------------\n";
  //reverse matches only make sense if we are checking similarity, otherwise it should be symmetric
  if(options.forward_reverse && !(options.do_simple_match) ) {
    //do the same in reverse and then cleanup the forward ones that are not in reverse
    for (auto itr=to_features.begin(); itr!=to_features.end(); itr++) {
      double top_score=INT_MAX, second_score=INT_MAX;
      //std::cerr<<"Vertex "<<itr->first<<": ";
      std::vector<RankFeature> closest_list;
      for (auto jtr=from_features.begin(); jtr!=from_features.end(); jtr++) {
        double dist = itr->second->compare(jtr->second);
        //keep track of best and second best match
        if(options.match_n_closest) {
          //we need to maintain a list and sort it
          closest_list.push_back(RankFeature(dist,jtr->second));
        } else {
          if(dist<top_score) {
            second_score = top_score;
            top_score = dist;
            match = jtr->second;
          } else if(dist<second_score) {
            second_score = dist;
          }
        }
      }
      if(options.match_n_closest) {
        //sort list
        std::sort(closest_list.begin(), closest_list.end(), [](RankFeature a, RankFeature b) 
            { return a.first < b.first; });
        for(int m=0; m<options.n_closest && m<closest_list.size(); m++) {
          //std::cerr<<"top "<<closest_list[0].first<<" current "<<closest_list[m].first<<std::endl;

          FeatureBasePtr vertex = itr->second;
          reverse.push_back(std::pair<FeatureBasePtr,FeatureBasePtr>(closest_list[m].second, vertex));
        }
      } else {
        if((top_score+1e-10)/(second_score+1e-10) < options.second_closest_thresh) {
          //std::cerr<<"top "<<top_score<<" second "<<second_score<<" ratio "<<(top_score+1e-10)/(second_score+1e-10)<<" match "<<match->id<<" to "<<itr->second->id<<std::endl;
          //we have a winner
          FeatureBasePtr vertex = itr->second;
          reverse.push_back(std::pair<FeatureBasePtr, FeatureBasePtr>(match, vertex));
        }
      }
    }

    std::cerr<<"Forward matches "<<forward.size()<<" reverse matches "<<reverse.size()<<std::endl;
    int process=0;
    //go through forward and look for the same match in reverese
    for(auto frw=forward.begin(); frw!=forward.end(); ++frw) {
      process++;
      auto rvrs=reverse.begin();
      //std::cerr<<"frwr "<<frw->first->id<<"-"<<frw->second->id;
      while(rvrs!=reverse.end()) {
        if(frw->first->id == rvrs->first->id && frw->second->id == rvrs->second->id) break;
        rvrs++;
      }
      if(rvrs != reverse.end()) {
        //found a match, all is well
        //std::cerr<<" rvrs "<<rvrs->first->id<<"-"<<rvrs->second->id<<std::endl;
        continue;
      } else {
        //std::cerr<<" no match\n";
        //no match
        forward.erase(frw);
        frw--; //go one back so we ++ correctly
      }
    }
        
    forward_vertices.clear();
    reverse_vertices.clear();
    for(auto frw=forward.begin(); frw!=forward.end(); ++frw) {
      forward_vertices.insert(frw->first->id);
      reverse_vertices.insert(frw->second->id);
    } 

    std::cerr<<"Matches in both are "<<forward.size()<<std::endl;
  }

  if(forward.size() < 3) {
    std::cerr<<"Not enough matches to RANSAC, only "<<forward.size()<<std::endl;
    return matches;
  } else {
    if(forward.size() == 3) {
      std::cerr<<"Unique transform\n";
      std::vector<Eigen::Vector3d> fromV, toV;
      for(int q=0; q<3; q++) {
        fromV.push_back(forward[q].first->pos);
        toV.push_back(forward[q].second->pos);
      }
      this->getTransform(fromV,toV, fromTo);
      for(auto mtr=forward.begin(); mtr!=forward.end(); mtr++) {
        matches.push_back(std::pair<int,int>(mtr->first->id, mtr->second->id));
      }
      return matches;
    }
    std::cerr<<"Searching through "<<forward.size()<<" matches\n";
  }

  fromTo = Eigen::Affine3d::Identity();
 
  double score = INT_MIN;
  if (options.random_seed)
    std::srand(static_cast<unsigned int>(std::time(nullptr))); //set random seed 
  else  
    std::srand(options.seed); //set seed 

  Eigen::Affine3d T;
  //This is actually 2*number of possible combinations
  int max_combos = forward.size()*(forward.size()-1)*(forward.size()-2)/3;
  max_combos = max_combos < options.max_iterations ? max_combos : options.max_iterations;
  std::cerr<<"max combinations = "<<max_combos<<std::endl;

  for(int i=0; i<max_combos; i++) {
    std::vector<Eigen::Vector3d> fromV, toV;
    /*
    for(int q=0; q<3; q++) {
      int idx = forward.size()*((double)std::rand()/RAND_MAX);
      //std::cerr<<"idx "<<idx<<" out of "<<forward.size()<<std::endl;
      fromV.push_back(forward[idx].first->pos);
      toV.push_back(forward[idx].second->pos);
    }
    */
    while(fromV.size()<3) {
      int idx = forward.size()*((double)std::rand()/RAND_MAX);
      bool insert=true;
      //check we haven't chosen a close-by point already
      for(int q=0; q<fromV.size(); q++) {
        insert = insert && ((forward[idx].first->pos-fromV[q]).norm()>0.1);
      }
      if(insert) {
        fromV.push_back(forward[idx].first->pos);
        toV.push_back(forward[idx].second->pos);
      }
      /* 
      else {
        std::cerr<<"jump point! "<<idx<<" = "<<forward[idx].first->pos.transpose()<<"\n already in are:\n";
        for(int q=0; q<fromV.size(); q++) {
          std::cerr<<fromV[q].transpose()<<std::endl;
        }
      }
      */
    }
    this->getTransform(fromV,toV, T);

    //score match;
    std::map<int,int> inlier_map, reverse_inlier_map;
    for(auto itr=forward.begin(); itr!=forward.end(); itr++) {
      double error = (T*itr->first->pos - itr->second->pos).norm();
      if(error<options.match_tolerance) {
        inlier_map[itr->first->id] = itr->second->id;
        reverse_inlier_map[itr->second->id] = itr->first->id;
      }
    }
    double sc = ((double)inlier_map.size()+reverse_inlier_map.size())/(forward_vertices.size()+reverse_vertices.size());
    if(sc>score) {
      std::cerr<<"Found a better transfrom with inlier ratio "<<sc<<std::endl;
      score=sc;
      fromTo = T;

      //Try to ICP closer
      if(options.refine_ICP) {
        std::cerr<<"Attempting to refine transform ...";
        fromV.clear(); toV.clear();
        //get new correspondences
        for(auto inliers=inlier_map.begin(); inliers!=inlier_map.end(); ++inliers) {
          FeatureBasePtr f,t;
          f = from_features[inliers->first]; 
          t = to_features[inliers->second];
          if(f!=nullptr && t!=nullptr) {
            fromV.push_back(f->pos);
            toV.push_back(t->pos);
          }
        }
        //get new transform
        if(fromV.size()<3) {
          std::cerr<<"not enough points\n";
          continue;
        }
        this->getTransform(fromV,toV, T);

        //score match;
        std::map<int,int> inlier_map_2, reverse_inlier_map_2;
        for(auto itr=forward.begin(); itr!=forward.end(); itr++) {
          double error = (T*itr->first->pos - itr->second->pos).norm();
          if(error<options.match_tolerance) {
            inlier_map_2[itr->first->id] = itr->second->id;
            reverse_inlier_map_2[itr->second->id] = itr->first->id;
          }
        }
        sc = ((double)inlier_map_2.size()+reverse_inlier_map_2.size())/(forward_vertices.size()+reverse_vertices.size());
        if(sc>score) {
          std::cerr<<" success, new score is "<<sc<<std::endl;
          score=sc;
          fromTo = T;
        } else {
          std::cerr<<" fail, new score is "<<sc<<std::endl;
        }
      }
    }
  }
  
  
  
  for(auto mtr=forward.begin(); mtr!=forward.end(); mtr++) {
    matches.push_back(std::pair<int,int>(mtr->first->id, mtr->second->id));
  }
  return matches;
}
  
