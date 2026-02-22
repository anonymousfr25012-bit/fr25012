#include <boomer2_tools/Detector.hh>

using namespace graph_matcher;


void FakeDetector::detect(cv::Mat &intensity, cv::Mat &depth, std::vector<cv::Point> &keypoints) {

  CamParams cp = rend_->getParams();

  for(int p=0; p<labels_.points.size(); p++) {
    Eigen::Vector3d point, point_in_cam;
    point<<labels_.points[p].x,labels_.points[p].y,labels_.points[p].z;
    //calculate point coordinates relative to camera
    point_in_cam = pose_.rotation().transpose()*(point-pose_.translation());
    //std::cerr<<"Point = "<<point.transpose()<<" in cam = "<<point_in_cam.transpose()<<std::endl;

    if(point_in_cam(2) <= 0.0) {
      continue; //point behind camera plane
    }

    cv::Point pt;
    rend_->project(pt, point_in_cam);

    //check if in FOV
    if(pt.x<0 || pt.x >= cp.image_x ||
       pt.y<0 || pt.y >= cp.image_y ) {
      continue;
    }
    //std::cerr<<"pixel "<<x<<" "<<y<<std::endl;
    keypoints.push_back(pt);
  }

}


void FastrcnnDetector::detect(cv::Mat &intensity, cv::Mat &depth, std::vector<cv::Point> &keypoints) {

  CamParams cp = rend_->getParams();

  // get labels using fasterrcnn
  getLabels(intensity, depth);
  // std::cerr<<labels_ <<std::endl;
  for(int p=0; p<labels_.size(); p++) {    
    cv::Point pt = labels_[p];    

    //check if in FOV
    if(pt.x<0 || pt.x >= cp.image_x ||
       pt.y<0 || pt.y >= cp.image_y ) {
      continue;
    }
    //std::cerr<<"pixel "<<x<<" "<<y<<std::endl;
    keypoints.push_back(pt);
  }

}


int FastrcnnDetector::getLabels(cv::Mat &intensity, cv::Mat &depth) {
  auto targetSize = intensity.size(); 
  if (intensity.size() != depth.size()) {
    std::cerr << "intensity and depth images have different sizes, use intensity size\n";
    cv::resize(depth, depth, targetSize);
  }
  intensity.convertTo(intensity, CV_32FC1);
  // intensity /= 1000.;
  depth.convertTo(depth, CV_32FC1);
  // depth /= 15000.;

  cv::Mat zeroMat = cv::Mat::zeros(targetSize, CV_32FC1);

  if (depth.size() != intensity.size() || depth.size() != zeroMat.size()) {
      std::cerr << "Error: Images must have the same size for merging." << std::endl;      
      return -1;
  }
  cv::Mat mergedImage;
  cv::Mat channels[] = {intensity, depth, zeroMat};
  cv::merge(channels, 3, mergedImage);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tensor_img = torch::from_blob(mergedImage.data, { mergedImage.rows, mergedImage.cols, mergedImage.channels() }, options);        
  tensor_img = tensor_img.permute({2, 0, 1});

  std::vector<torch::Tensor> images;
  images.push_back(tensor_img.to(device_));

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(images);

  auto outputs = module_.forward(inputs).toTuple()->elements();

  auto val_out = outputs[1].toList().get(0).toGenericDict();
  auto boxes = val_out.at("boxes");
  // auto labels = val_out.at("labels");
  auto scores = val_out.at("scores");

  at::Tensor bbox = boxes.toTensor().to("cpu").data();
  at::Tensor bscore = scores.toTensor().to("cpu").data();
  std::vector<float> scoresVec(bscore.data<float>(), bscore.data<float>() + bscore.numel()); // tensor.data<float>() provides a pointer to the data in the tensor; tensor.numel() returns the total number of elements in the tensor

  // Print the elements of the std::vector
  // std::cout << "Converted std::vector elements: ";
  // for (const auto& score : scoresVec) {
  //     std::cout << score << " ";
  // }

  // std::cout << std::endl;

  // Get the sizes of the tensor
  std::vector<int64_t> sizes = bbox.sizes().vec();

  // Convert the tensor to a nested std::vector
  labels_.clear();
  labels_.resize(0);
  labels_.reserve(sizes[0]);

  std::vector<cv::Rect> boxesVector;
  boxesVector.reserve(sizes[0]);

  cv::Mat rgbImage;
  cv::cvtColor(intensity, rgbImage, cv::COLOR_GRAY2RGB);
  for (int64_t i = 0; i < sizes[0]; ++i) {
      cv::Rect rect = tensorToRect(bbox[i]);
      cv::Point center(rect.y + rect.height / 2, rect.x + rect.width / 2);
      const auto box = rect;
      const auto score = scoresVec[i];
      if (box.area() > 0 && score > 0.6) {  // Check if the bounding box is non-empty
          cv::rectangle(rgbImage, box, cv::Scalar(0, 0, 255), 1);
          // cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
          
          boxesVector.push_back(rect);
          labels_.push_back(center);
      }
     
  }

  // Print the elements of the nested std::vector
  // std::cout << "Converted boxes std::vector elements:\n";
  // for (const auto& row : boxesVector) {
  //   std::cout << row << '\n';
  // }

  // std::cout << std::endl;

  // draw results
  // Filter predictions to show non-overlapping bounding boxes with score > 0.5
  // for (int64_t i = 0; i < sizes[0]; ++i) {
  //   const auto box = boxesVector[i];
  //   const auto score = scoresVec[i];
  //     if (box.area() > 0 && score > 0.8) {  // Check if the bounding box is non-empty
  //         cv::rectangle(rgbImage, box, cv::Scalar(0, 0, 255), 1);
  //     }
  // }

  // get features

  // fix me
  // std::vector<torch::jit::IValue> stack;
  // auto exported_method = module_.get_method("get_modules");
  // auto boxes_features = exported_method(stack);
  // auto features = boxes_features.toList().get(0).toTensor().to("cpu");

  // // Print the resized matrix
  // std::vector<float> boxes_features_f(features.data<float>(), features.data<float>() + features.numel());
  // std::vector<double> boxes_features_d;
  // for (auto const a : boxes_features_f) {
  //     boxes_features_d.push_back(static_cast<float>(a));
  // }
  // int num_boxes = boxes_features_d.size()/1024;
  // this->features_ = Eigen::Map<Eigen::MatrixXd>(boxes_features_d.data(), num_boxes, 1024);


  // cv::imshow("Detected results", rgbImage);  
  // cv::imshow("original image intensity", intensity);
  // cv::imshow("original image depth", depth); 
  return 0;
}

int FastrcnnDetector::setModel(const std::string model_path, const bool useCPU=false) {
  try
  {

    const bool cuda_is_available = useCPU ? false : torch::cuda::is_available();
    device_ = cuda_is_available ? torch::kCUDA : torch::kCPU;
    std::cout << "cuda is available: " << cuda_is_available << std::endl;
    std::string device_name = cuda_is_available ? "CUDA" : "CPU";
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(model_path, device_);    
    module_.eval();
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "model loaded!\n";
  return 0;
}

void FastrcnnDetector::getFeatures(Eigen::MatrixXd &features) {
   features = this->features_;
}

cv::Rect FastrcnnDetector::tensorToRect(const at::Tensor& tensor) {
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