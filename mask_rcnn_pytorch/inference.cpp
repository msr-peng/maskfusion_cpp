#include "inference.h"

namespace fs = std::experimental::filesystem;

void inference(std::vector<ClassColour> &colour_scheme, const cv::Mat &input_image, cv::Mat &output_image, cv::Mat& scores_matrix, cv::Mat& class_ids_matrix, MaskRCNN& model, std::shared_ptr<InferenceConfig> config, bool output_flag) {
  try {
    std::vector<cv::Mat> images{input_image};
    // Mold inputs to format expected by the neural network
    at::Tensor molded_images;
    std::vector<ImageMeta> image_metas;
    std::vector<Window> windows;
    std::tie(molded_images, image_metas, windows) =
	MoldInputs(images, *config.get());

    at::Tensor detections, mrcnn_mask;
    std::tie(detections, mrcnn_mask) = model->Detect(molded_images, image_metas);
    if (!is_empty(detections)) {
      // Process detections
      //[final_rois, final_class_ids, final_scores, final_masks]
      using Result =
          std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<cv::Mat>>;
      std::vector<Result> results;

      double mask_threshold = 0.5;
      for (size_t i = 0; i < images.size(); ++i) {
        auto result =
            UnmoldDetections(detections[static_cast<int64_t>(i)],
                             mrcnn_mask[static_cast<int64_t>(i)], input_image.size(),
                             windows[i], mask_threshold);
        results.push_back(result);
      }

      float score_threshold = 0.7f;
      visualize(colour_scheme, input_image, output_image, scores_matrix, class_ids_matrix, std::get<0>(results[0]), std::get<1>(results[0]),
                std::get<2>(results[0]), std::get<3>(results[0]),
                score_threshold, GetDatasetClasses(), output_flag);
    } else {
      std::cerr << "Failed to detect anything!\n";
    }

  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
  }
}
