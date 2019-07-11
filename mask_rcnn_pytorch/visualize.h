#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../src/utilities/Types.h"

void output(string file_name, cv::Mat &output_mat);

void visualize(std::vector<ClassColour> &colour_scheme,
               const cv::Mat& input_image,
	       cv::Mat& output_image,
	       cv::Mat& scores_matrix,
	       cv::Mat& class_ids_matrix,
               at::Tensor boxes,
               at::Tensor class_ids,
               at::Tensor scores,
               const std::vector<cv::Mat>& masks,
               float score_threshold,
               const std::vector<std::string>& class_names,
	       bool output_flag = false);

#endif  // VISUALIZE_H
