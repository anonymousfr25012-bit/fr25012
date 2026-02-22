#include <boomer2_tools/CloudRenderer.hh>
#include <boomer2_tools/GraphBuilder.hh>
#include <boomer2_tools/Detector.hh>
#include <boomer2_tools/Descriptor.hh>
#include <boomer2_tools/Logger.hh>

using namespace graph_matcher;

int main(int argc, char **argv)
{

  FeatureGraphPtr graph = std::make_shared<FeatureGraph>();
  graph->load("first.graph");
  FeatureGraphPtr graph_second = std::make_shared<FeatureGraph>();
  graph_second->load("second.graph");


  graph_matcher::CamParams cp;
  std::shared_ptr<graph_matcher::CloudRenderer> rend(new graph_matcher::CloudRenderer(cp));
  graph_matcher::GraphBuilder gb(rend);
          
  //------------RANSAC options--------------/
  graph_matcher::RANSAC_options options;
  options.match_tolerance = 0.1;
  options.max_iterations = 1000000;
  options.seed = std::time(0);
  options.refine_ICP = false;
  options.forward_reverse = false;
  options.do_simple_match = false;
  options.match_n_closest = true;
  options.dist_threshold = 0.1;        // if doing simple match, consider matches with a score below this
  options.second_closest_thresh = 0.9; // >=1 means just match to closest
  options.n_closest = 3;               // if doing closest n, these many n

  Eigen::Affine3d reg;

  auto matches = gb.matchGraphs(graph, graph_second, options, reg);

  std::cerr << "Got transform " << reg.matrix() << std::endl;

  return 0;
}
