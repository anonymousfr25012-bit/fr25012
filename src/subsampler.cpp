/**
  * Simple utility to subsample a scan given as a pcd file
  * Expects the voxel grid resolution as a parameter
  */
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <boost/program_options.hpp> 
namespace po = boost::program_options;

int main(int argc, char** argv) {

  std::string in_file, out_file;
  double grid_resolution;
  po::options_description desc("Allowed options");
  desc.add_options() 	
    ("help", "produce help message") 	
    ("input", po::value<std::string>(&in_file), "name of the input pcd file") 	
    ("output", po::value<std::string>(&out_file), "name of the output pcd file") 	
    ("resolution", po::value<double>(&grid_resolution)->default_value(0.1), "size of the grid to use")
  ;

  po::variables_map vm;     
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))     
  { 	
    std::cout << desc << "\n"; 	
    return 1;     
  }
  if (!vm.count("input") || !vm.count("output"))     
  {  
    std::cout << "Please provide an input and output file name tags. Run with --help for usage.\n";	
    return -1;     
  }

  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
  pcl::PCDReader reader;

  if(reader.read(in_file, *cloud) < 0) {
    std::cerr<<"Error reading pcd file\n";
    return -1;
  }

  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
    << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize(grid_resolution, grid_resolution, grid_resolution);
  sor.filter (*cloud_filtered);

   std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
     << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

   pcl::PCDWriter writer;
   writer.write (out_file, *cloud_filtered);

   return 0;
}
