#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/pcl_config.h>

#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Geometry>

#include <sstream> // std::stringstream
#include <regex> // std::regex
#include <filesystem> // std::filesystem

#include <boomer2_tools/CloudRenderer.hh>
#include <boomer2_tools/GraphBuilder.hh>
#include <boomer2_tools/Detector.hh>
#include <boomer2_tools/Descriptor.hh>
#include <boomer2_tools/Logger.hh>


#include "omp.h"

namespace po = boost::program_options;


class Graph_Info {
    public:
        Graph_Info(const std::string& cur_graph, const std::string& prev_graph, const std::string& base_graph, int ref_counter, const Eigen::Affine3d& tf_wrt_prev, const Eigen::Affine3d& tf, const Eigen::Affine3d& tf_gt)
            : cur_graph(cur_graph), prev_graph(prev_graph), base_graph(base_graph), ref_counter(ref_counter), tf_wrt_prev(tf_wrt_prev), tf(tf), tf_gt(tf_gt) {
            update_error();
        }

        std::string cur_graph;
        std::string prev_graph;
        std::string base_graph;
        int ref_counter;
        Eigen::Affine3d tf_wrt_prev;
        Eigen::Affine3d tf;
        Eigen::Affine3d tf_gt;
        double err_angle;
        double err_trans;

        void update_error() {
            // calculate translation error
            this->err_trans = (tf.translation() - tf_gt.translation()).norm();

            // calculate rotation error
            Eigen::Matrix3d rel_rotation = tf.rotation().transpose() * tf_gt.rotation();
            // get angle axis    
            double angle = std::acos((rel_rotation.trace() - 1) / 2);
            
            this->err_angle = angle * 180.0 / M_PI;
        }

        std::string to_string() const {
            std::ostringstream oss;
            oss << "Current: " << cur_graph << ", Prev: " << prev_graph << ", Base: " << base_graph << ",\n"
                << "  Ref count: " << ref_counter << ", Rot error: " << err_angle << ", Tr error: " << err_trans;
            return oss.str();
        }
};

bool find_absolute_transformation(const std::string& cur_file, const std::map<std::string, std::string>& graph_relations, const std::map<std::string, Eigen::Affine3d>& reg_matrixes, const std::string& base_file, Eigen::Affine3d& transformation_matrix, int& ref_counter) {
    transformation_matrix = Eigen::Affine3d::Identity();
    ref_counter = 0;
    std::string current = cur_file;

    if (cur_file == base_file) {
        return true;
    }

    while (current != base_file) {
        ref_counter++;
        auto it = graph_relations.find(current);
        if (it == graph_relations.end()) {
            throw std::runtime_error("Base file not found in graph relations.");
            return false;
        }
        const std::string& next_file = it->second;
        transformation_matrix = reg_matrixes.at(current) * transformation_matrix;
        current = next_file;
    }

    return true;
}

Eigen::Affine3d extract_abs_tf_from_filename(const std::string& filename) {
    std::regex pattern(R"((-?\d+)_(-?\d+)_(-?\d+))");
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        int x = std::stoi(match[1].str());
        int y = std::stoi(match[2].str());
        int z = std::stoi(match[3].str());

        Eigen::Affine3d transformation_matrix = Eigen::Affine3d::Identity();
        transformation_matrix.translation() << x, -z, y;
        return transformation_matrix;
    } else {
        throw std::runtime_error("Filename does not match the expected pattern.");
    }
}

int main(int argc, char **argv)
{ 
    std::string graph_folder, base_graph;
    bool debug = false;
    bool load_pts = false;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("debug", "print debug information")
    ("load_pts", "load point clouds")
    ("graphs", po::value<std::string>(&graph_folder),"folder which contain graphs and their relations")
    ("base", po::value<std::string>(&base_graph)->default_value("18_-3_-19"), "base graph file")
    
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);


    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("debug")) {
        debug = true;
    }

    if (vm.count("load_pts")) {
        load_pts = true;
    }

    if (!vm.count("graphs")) {
        std::cout << "Please provide the folder which contain graphs and their relations. Run with --help for usage.\n";
        return -1;
    }

    if (!vm.count("base"))
    {
        std::cout << "No base graph name provided, by default " << base_graph << " is used.\n";
        return -1;
    }
    

    std::cout << "Graph folder: " << graph_folder << "\n";

    // Load the graph files and their relations, and store them in a dictionary
    // The graph files are named as reg_matrix_(input_2)_(input_1).txt, input_* is the matcher input files    

    std::vector<std::string> graph_files;
    std::map<std::string, std::string> graph_dict;
    std::map<std::string, Eigen::Affine3d> graph_transforms;
    for (const auto & entry : std::filesystem::directory_iterator(graph_folder)) {
        const std::string file_name = entry.path().filename().string();
        std::regex re(R"reg(reg_matrix_icp_\(([^)]+)\)_\(([^)]+)\)\.txt)reg");
        std::smatch match;
        if (std::regex_match(file_name, match, re) ) {
            std::string a = match[1].str();
            std::string b = match[2].str();
            graph_files.push_back(file_name);
            std::cout << "Found graph file: " << file_name << "; " << a << " : " << b << "\n";
            graph_dict[a] = b;
            
            // read matrix from file
            std::ifstream file(entry.path());
            if (file.is_open()) {
                Eigen::Affine3d transform;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        file >> transform(i, j);
                    }
                }
                graph_transforms[a] = transform;
                if (debug) {
                    std::cout << "Transform matrix for " << a << " : " << b << "\n" << transform.matrix() << "\n";
                }
            } else {
                std::cerr << "Failed to open file: " << entry.path() << "\n";
            }

            
        }
    }

    // Print the graph dictionary
    std::cout << "Graph dictionary:\n";
    for (const auto & pair : graph_dict) {
        std::cout << pair.first << " -> " << pair.second << "\n";
    }

    std::vector<Graph_Info> graph_infos;
    // get transofrmation matrix for each graph
    for (const auto & pair : graph_dict) {
        Eigen::Affine3d transform;
        int ref_counter;
        if (find_absolute_transformation(pair.first, graph_dict, graph_transforms, base_graph, transform, ref_counter)) {
            std::cout << "Transformation matrix for " << pair.first << " wrt base:\n" << transform.matrix() << "\n";
            std::cout << "Reference counter: " << ref_counter << "\n";
            Eigen::Affine3d transform_gt = extract_abs_tf_from_filename(pair.first);
            auto graph = Graph_Info(pair.first, pair.second, base_graph, ref_counter, graph_transforms[pair.first], transform, transform_gt);
            graph_infos.push_back(graph);
        } else {
            std::cerr << "Failed to find transformation matrix for " << pair.first << "\n";
        }

        
    }
    // load the graph files
    std::map<std::string, graph_matcher::FeatureGraphPtr> graph_map;
    std::map<std::string, graph_matcher::FeatureGraphPtr> feature_map;
    for (const auto & pair : graph_dict) {
        std::string key = pair.first;
        std::string file_name = key + ".graph";
        std::string file_path = graph_folder + file_name;
        graph_matcher::FeatureGraphPtr feature(new graph_matcher::FeatureGraph());
        if (!feature->load(file_path)) {
            std::cerr << "Failed to load feature file: " << file_path << "\n";
            return -1;
        }
        std::cout << "Loaded feature file: " << file_path << "\n";
        feature_map[key] = feature;
    }

    // test 2 graph  18_-3_-19   13_-3_-20 
    auto graph_1 = std::string("18_-3_-19");
    auto graph_2 = std::string("13_-3_-20");
    auto graph_1_feature = feature_map[graph_1];
    auto graph_2_feature = feature_map[graph_2];
    // connect all the graphs
    

}