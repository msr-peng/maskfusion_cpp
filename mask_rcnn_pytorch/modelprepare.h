#ifndef MODELPREPARE_H
#define MODELPREPARE_H

#include "config.h"
#include "maskrcnn.h"
#include "stateloader.h"

#include <experimental/filesystem>
#include <iostream>
#include <tuple>

class InferenceConfig : public Config {
 public:
  InferenceConfig() {
    if (!torch::cuda::is_available())
      throw std::runtime_error("Cuda is not available");
    gpu_count = 1;
    images_per_gpu = 1;
    num_classes = 81;  // 4 - for shapes, 81 - for coco dataset

    UpdateSettings();
  }
};

std::tuple<MaskRCNN, std::shared_ptr<InferenceConfig>> model_prepare(const std::string &model_path);

#endif  // MODELPREPARE_H
