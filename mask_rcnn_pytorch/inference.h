#ifndef INFERENCE_H
#define INFERENCE_H

#include "cocoloader.h"
#include "config.h"
#include "datasetclasses.h"
#include "debug.h"
#include "imageutils.h"
#include "maskrcnn.h"
#include "stateloader.h"
#include "visualize.h"
#include "modelprepare.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <memory>

void inference(const cv::Mat &input_image, cv::Mat &output_image, cv::Mat& scores_matrix, cv::Mat& class_ids_matrix, MaskRCNN& model, std::shared_ptr<InferenceConfig> config, bool output_flag = false);

#endif  // INFERENCE_H
