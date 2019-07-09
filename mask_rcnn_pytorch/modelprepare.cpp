#include "modelprepare.h"

namespace fs = std::experimental::filesystem;

std::tuple<MaskRCNN, std::shared_ptr<InferenceConfig>> model_prepare(const string& model_path) {
  std::string params_path = fs::canonical(model_path);
  if (!fs::exists(params_path))
    throw std::invalid_argument("Wrong file path for MaskRCNN model");

  auto config = std::make_shared<InferenceConfig>();

  // Root directory of the project
  auto root_dir = fs::current_path();
  // Directory to save logs and trained model
  auto model_dir = root_dir / "logs";
  
  // Create model object.
  MaskRCNN model(model_dir, config);
  
  // load state before moving to GPU
  if (params_path.find(".json") != std::string::npos) {
    LoadStateDictJson(*model, params_path);
  } else {
      LoadStateDict(*model, params_path, "");
  }
  
  if (config->gpu_count > 0)
    model->to(torch::DeviceType::CUDA);
  
  return std::make_tuple(model, config);
}