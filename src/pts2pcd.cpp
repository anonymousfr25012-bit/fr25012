#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char**argv) {

  if(argc<3) {
    cerr<<"Usage: ./"<<argv[0]<<" input_pts_file output_pcd_file\n";
    return -1;
  }

  string in_file = argv[1];
  string out_file = argv[2];
  float resolution = 0;
  float max_distance = 0;
  if (argc == 4) {
    try {
      resolution = stof(argv[3]);
    } catch (const invalid_argument& e) {
      cerr << "Invalid resolution value provided. Please provide a valid float.\n";
      return -4;
    } catch (const out_of_range& e) {
      cerr << "Resolution value out of range. Please provide a valid float.\n";
      return -5;
    }
  }

  if (argc == 5) {
    try {
      max_distance = stof(argv[4]);
    } catch (const invalid_argument& e) {
      cerr << "Invalid max_distance value provided. Please provide a valid float.\n";
      return -6;
    } catch (const out_of_range& e) {
      cerr << "Max_distance value out of range. Please provide a valid float.\n";
      return -7;
    }
  }

  ifstream instream(in_file, std::ios::in);
  if(!instream.is_open()) {
    cerr<<"Could not open file "<<in_file<<" for reading. Please check path\n";
    return -2;
  }
  if(!instream.good()) {
    cerr<<"File "<<in_file<<" cannot be read.\n";
    return -3;
  }

  int n_points = 0;
  instream >> n_points;
  cout<<"Reading from file "<<in_file<<" with "<<n_points<<" points\n";

  //expected format is: x y z intensity rgb_r rgb_g rgb_b
  pcl::PointXYZI pt;
  pcl::PointCloud<pcl::PointXYZI> cloud;

  int r,g,b,i;

  while(instream.good()) {
    instream>>pt.x>>pt.y>>pt.z>>i>>r>>g>>b;
    if (max_distance > 0) {
      if (pt.getVector3fMap().norm() > max_distance) {
        continue;
      }
    }
    //cout<<"Read "<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<i<<" "<<r<<" "<<g<<" "<<b<<endl;
    pt.intensity = (r+g+b)/765.; //average RGB value
    cloud.points.push_back(pt);
  }

  cout<<"Read "<<cloud.points.size()<<" points\n";
  cloud.is_dense=false;
  cloud.width=1;
  cloud.height=cloud.points.size();

  std::cerr << "PointCloud before filtering: " << cloud.width * cloud.height
    << " data points." << std::endl;

  pcl::PointCloud<pcl::PointXYZI> cloud_filtered;

  
  if (resolution > 0) {
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud.makeShared());
    sor.setLeafSize(resolution, resolution, resolution);
    sor.filter(cloud_filtered);
  }

  cout << "Filtered cloud contains " << cloud_filtered.points.size() << " points\n";
  pcl::PCDWriter writer;
  writer.writeBinary(out_file,cloud_filtered);

  return 0;
}
