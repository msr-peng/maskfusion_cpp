#include "visualize.h"
#include "debug.h"

#include <string>
#include <vector>
#include <stdio.h>

void output(string file_name, cv::Mat &output_mat) {
  std::ofstream output_file;
  output_file.open(file_name);
  /*
  for (int y = 0; y < (output_mat.rows/10); ++y) {
    uchar *row_ptr = output_mat.ptr<uchar>(10*y);
    for (int x = 0; x < (output_mat.cols/10); ++x) {
      int val = row_ptr[10*x];
      output_file << val;
    }
    output_file << "\n";
  }
  */
  float *ptr = (float*)output_mat.data;
  for (int y = 0; y < (output_mat.rows/10); ++y) {
    for (int x = 0; x < (output_mat.cols/10); ++x) {
      float val = ptr[640*10*y + 10*x];
      output_file << val << " ";
    }
    output_file << "\n";
  }
  output_file.close();
}

void visualize(const cv::Mat& image,
	       cv::Mat& output_image,
	       cv::Mat& scores_matrix,
	       cv::Mat& class_ids_matrix,
               at::Tensor boxes,
               at::Tensor class_ids,
               at::Tensor scores,
               const std::vector<cv::Mat>& masks,
               float score_threshold,
               const std::vector<std::string>& class_names,
	       bool output_flag) {
  cv::Mat img = image.clone();
  auto n = boxes.size(0);
  for (int64_t i = 0; i < n; ++i) {
    auto score = *scores[i].data<float>();
    if (score >= score_threshold) {
      auto bbox = boxes[i];
      auto y1 = *bbox[0].data<int32_t>();
      auto x1 = *bbox[1].data<int32_t>();
      auto y2 = *bbox[2].data<int32_t>();
      auto x2 = *bbox[3].data<int32_t>();
      auto class_id = *class_ids[i].data<int64_t>();

      cv::Mat bin_mask = masks[i].clone();
      cv::Mat mask_ch[3];
      mask_ch[2] = bin_mask;
      mask_ch[0] = cv::Mat::zeros(img.size(), CV_8UC1);
      mask_ch[1] = cv::Mat::zeros(img.size(), CV_8UC1);
      cv::Mat mask;
      cv::merge(mask_ch, 3, mask);
      cv::addWeighted(img, 1, mask, 0.5, 0, img);

      cv::Point tl(static_cast<int>(x1), static_cast<int>(y1));
      cv::Point br(static_cast<int>(x2), static_cast<int>(y2));
      cv::rectangle(img, tl, br, cv::Scalar(1, 0, 0));
      cv::putText(img,
                  class_names[static_cast<size_t>(class_id)] + " - " +
                      std::to_string(score),
                  cv::Point(tl.x + 5, tl.y + 5),   // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                  1.0,                             // Scale. 2.0 = 2x bigger
                  cv::Scalar(255, 100, 255));      // BGR Color

      // output for cuda kernel to update probabilities
      bin_mask.convertTo(bin_mask, CV_32F);
      bin_mask /= 255.0;
      if (output_flag) {
	std::string prefix = "/home/Downloads/outputs/mask_";
	std::string num = std::to_string(i);
	output(prefix + num, bin_mask);
	cv::imwrite(prefix + num + ".jpg", bin_mask);
      }
      cv::addWeighted(scores_matrix, 1, bin_mask, score, 0, scores_matrix);
      cv::addWeighted(class_ids_matrix, 1, bin_mask, class_id, 0, class_ids_matrix);
    }
  }
  output_image = img;
  cv::imwrite("result.png", img);
}