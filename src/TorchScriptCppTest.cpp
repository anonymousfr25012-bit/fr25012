#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torchvision/vision.h>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <Eigen/Core>

namespace po = boost::program_options;

cv::Rect tensorToRect(const at::Tensor& tensor) {
    // Ensure the tensor has the correct size
    TORCH_CHECK(tensor.ndimension() == 1 && tensor.size(0) == 4, "Invalid tensor size for bounding box");

    // Extract coordinates from the tensor
    int x = static_cast<int>(tensor[0].item<float>());
    int y = static_cast<int>(tensor[1].item<float>());
    int width = static_cast<int>(tensor[2].item<float>() - x);
    int height = static_cast<int>(tensor[3].item<float>() - y);

    // Create and return a cv::Rect
    return cv::Rect(x, y, width, height);
}

int test_model(int argc, const char *argv[]) {

  bool cuda_is_available;
  std::string in_file_depth, in_file_intensity, model_file;
 po::options_description desc("Allowed options");
  desc.add_options()
  ("help", "produce help message")
  ("cpu", "run libtorch with cpu")
  ("cuda", "run libtorch with cuda")
  ("get_features", "get features")
  ("input_depth", po::value<std::string>(&in_file_depth), "name of the input depth image")
  ("input_intensity", po::value<std::string>(&in_file_intensity), "name of the input intensity image")
  ("model", po::value<std::string>(&model_file), "file path of the model")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("cpu") + vm.count("cuda") == 0)  {
    cuda_is_available = torch::cuda::is_available();
  } else if (vm.count("cpu")) {
    cuda_is_available = false;
  } else if (vm.count("cuda")) {
    cuda_is_available = torch::cuda::is_available();
  } else {
    cuda_is_available = torch::cuda::is_available();
  }
  
  bool get_features = false;
  if (vm.count("get_features")) {
    get_features = true;
  }

  bool is_real_img = false;
  std::string img_path_depth, img_path_intensity;
  if (vm.count("input_depth") && + vm.count("input_intensity")) {
    is_real_img = true;
    std::cerr << "given an depth image at: " << in_file_depth << std::endl;  
    std::cerr << "given an intensity image at: " << in_file_intensity << std::endl;  
    img_path_depth = in_file_depth;
    img_path_intensity = in_file_intensity;
  } else {
    std::cerr << "use fake image" << std::endl;
  }

  if (vm.count("model") == 0) {
    std::cerr << "need a model, please run with --model /path/to/your/model/xxx.pt" << std::endl;
  }

  auto device = cuda_is_available ? torch::kCUDA : torch::kCPU;  
  std::cout << "cuda is available: " << cuda_is_available << std::endl;
  std::string device_name = cuda_is_available ? "CUDA" : "CPU";

  torch::jit::script::Module module;

  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_file, device);
    // module.to(at::kCUDA);
    module.eval();
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "model loaded!\n";

  

  int img_cols = 512;
  int img_rows = 512;


  std::vector<torch::Tensor> images;
  // load image if available
  std::cerr << img_path_depth.empty() << img_path_depth.empty() << std::endl;

  cv::Mat img_depth;
  cv::Mat img_intensity;
  torch::Tensor tensor_img;
  if (is_real_img) {
    img_depth = cv::imread(img_path_depth, cv::IMREAD_UNCHANGED);
    img_intensity = cv::imread(img_path_intensity, cv::IMREAD_UNCHANGED);

    // Check if the images are loaded successfully
    if (img_depth.empty()) {
        std::cerr << "Error loading depth images." << std::endl;
        return -1;
    }

    if (img_intensity.empty()) {
        std::cerr << "Error loading intensity images." << std::endl;
        return -1;
    }

    std::cerr << "Images loaded" << std::endl;
    cv::Size target_size(img_cols, img_rows);
    cv::resize(img_depth, img_depth, target_size);
    
    cv::resize(img_intensity, img_intensity, target_size);


    // Merge the two images and a zeromat into a 3-channel image
    cv::Mat zeroMat = cv::Mat::zeros(target_size, CV_32FC1);
    img_intensity.convertTo(img_intensity, CV_32FC1);
    img_intensity /= 1000.;
    img_depth.convertTo(img_depth, CV_32FC1);
    img_depth /= 15000.;

    std::cerr << "size: " 
              << "depth: " << img_depth.size() 
              << "intensity: " << img_intensity.size() 
              << "depth" << zeroMat.size() << std::endl;
    
    std::cerr << "depth: "
              << "depth: " << img_depth.depth() 
              << "intensity: " << img_intensity.depth() 
              << "depth: " << zeroMat.depth() << std::endl;
    // Check if all images have the same size and depth
    if (img_depth.size() != img_intensity.size() || img_depth.size() != zeroMat.size()) {
        std::cerr << "Error: Images must have the same size for merging." << std::endl;
        return -1;  // Or handle the error in an appropriate way
    }

    cv::Mat mergedImage;
    cv::Mat channels[] = {img_intensity, img_depth, zeroMat};
    cv::merge(channels, 3, mergedImage);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    tensor_img = torch::from_blob(mergedImage.data, { mergedImage.rows, mergedImage.cols, mergedImage.channels() }, options);        
    std::cout << "Image Type: " << mergedImage.type() << std::endl;
    
    // Find the minimum and maximum values
    torch::Tensor minVal = tensor_img.min();
    torch::Tensor maxVal = tensor_img.max();

    // Display the results
    std::cout << "Minimum Value: " << minVal.item<float>() << std::endl;
    std::cout << "Maximum Value: " << maxVal.item<float>() << std::endl;

    tensor_img = tensor_img.permute({2, 0, 1});

    // Print the size of the tensor
    std::cout << "Tensor Size: " << tensor_img.sizes() << std::endl;

    images.push_back(tensor_img.clone().to(device));
    
  } else { // test data
    //******************************************
    //******* input example for resnet *********
    //******************************************
    //   std::vector<torch::jit::IValue> inputs;
    //   inputs.push_back((torch::ones({1, 3, 512, 512})).to(at::kCUDA));

    //   // Execute the model and turn its output into a tensor.
    //   at::Tensor output = module.forward(inputs).toTensor();
    //   std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    // *****************************************
    
    //******************************************
    //****** input example for fasterrcnn ******
    //******************************************
    // Faster RCNN accepts a List[Tensor] as main input
    images.push_back(torch::rand({3, img_cols, img_rows}).to(device));
    images.push_back(torch::rand({3, img_cols, img_rows}).to(device));
  }


  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;
  inputs.clear();
  inputs.resize(0);
  inputs.push_back(images);

  

  std::cerr << "start prediction..." << std::endl;
  auto outputs = module.forward(inputs).toTuple()->elements();

  if (get_features){
    // try to get boxes with features
    std::cerr << "try to get features" << std::endl;
    std::vector<torch::jit::IValue> stack;
    auto exported_method = module.get_method("get_modules");
    auto boxes_features = exported_method(stack);
    auto features = boxes_features.toList().get(0).toTensor().to("cpu");
    std::cerr << "aaaaaaaaaaaa" << std::endl;

    // Print the resized matrix
    std::cerr << "bbbbbbbbbbbaaaaaa" << std::endl;
    std::vector<float> boxes_features_f(features.data<float>(), features.data<float>() + features.numel());
    std::vector<double> boxes_features_d;
    for (auto const a : boxes_features_f) {
        boxes_features_d.push_back(static_cast<float>(a));
    }
    int num_boxes = boxes_features_d.size()/1024;
    Eigen::MatrixXd resized_matrix = Eigen::Map<Eigen::MatrixXd>(boxes_features_d.data(), num_boxes, 1024);
    // for (auto a : boxes_features_v) std::cout << a << std::endl;

    std::cout << resized_matrix << std::endl;
    std::cerr << boxes_features_d.size() << typeid(boxes_features_d).name()  << std::endl;
    // std::cerr << typeid(boxes_features).name() << "\nfeatures:" << boxes_features << std::endl;
    // std::cerr << "backbone" << module.backbone<< std::endl;
  }
  
  auto val_out = outputs[1].toList().get(0).toGenericDict();
  auto boxes = val_out.at("boxes");
  // auto labels = val_out.at("labels");
  auto scores = val_out.at("scores");

  at::Tensor bbox = boxes.toTensor().data();
  at::Tensor bscore = scores.toTensor().data();
  

  std::cout << "bbox: " << bbox << "; bscore: " << bscore << std::endl;

  bbox = boxes.toTensor().to("cpu").data();
  bscore = scores.toTensor().to("cpu").data();
  // show image if the input is real image
  if (is_real_img) {
    // Convert the tensor to a std::vector
    std::vector<float> scoresVec(bscore.data<float>(), bscore.data<float>() + bscore.numel()); // tensor.data<float>() provides a pointer to the data in the tensor; tensor.numel() returns the total number of elements in the tensor

    // Print the elements of the std::vector
    std::cout << "Converted std::vector elements: ";
    for (const auto& score : scoresVec) {
        std::cout << score << " ";
    }

    std::cout << std::endl;

    // Get the sizes of the tensor
    std::vector<int64_t> sizes = bbox.sizes().vec();

    // Convert the tensor to a nested std::vector
    std::vector<cv::Rect> boxesVector;
    boxesVector.reserve(sizes[0]);

    for (int64_t i = 0; i < sizes[0]; ++i) {
        cv::Rect row = tensorToRect(bbox[i]);
        boxesVector.push_back(row);
    }

    // Print the elements of the nested std::vector
    std::cout << "Converted boxes std::vector elements:\n";
    for (const auto& row : boxesVector) {
      std::cout << row << '\n';
    }

    std::cout << std::endl;

    // draw results
    // Filter predictions to show non-overlapping bounding boxes with score > 0.5
    cv::Mat rgbImage;
    cv::cvtColor(img_intensity, rgbImage, cv::COLOR_GRAY2RGB);
    for (int64_t i = 0; i < sizes[0]; ++i) {
      const auto box = boxesVector[i];
      const auto score = scoresVec[i];
        if (box.area() > 0 && score > 0.5) {  // Check if the bounding box is non-empty
            cv::rectangle(rgbImage, box, cv::Scalar(0, 0, 255), 1);
        }
    }

    cv::imshow("Detected results", rgbImage);  
    cv::imshow("original image intensity", img_intensity);
    cv::imshow("original image depth", img_depth); 
    cv::waitKey();
  } 

  return 0;
}

int main(int argc, const char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "usage: ./testLibTorch <path-to-exported-script-module>\n";
    return -1;
  }  

  test_model(argc, argv);

  return 0;  

}
