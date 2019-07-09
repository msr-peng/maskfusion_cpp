/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "SemanticFusionInterface.h"
#include "SemanticFusionCuda.h"
#include <utilities/Stopwatch.h>
#include <set>
#include <cmath>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <unordered_set>
#include <utility>

#include <iostream>

namespace std 
{
  template <class T>
  inline void my_hash_combine(std::size_t & seed, const T & v)
  {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2); 
  }

  template<typename S, typename T> struct hash<pair<S, T>> 
  {
    inline size_t operator()(const pair<S, T> & v) const
    {   
      size_t seed = 0;
      my_hash_combine(seed, v.first);
      my_hash_combine(seed, v.second);
      return seed;
    }   
  };  
}

// Basic algorithm for removing old items from probability table
template<typename T>
void remove_index(std::vector<T>& vector, const std::vector<int>& to_remove) {
  auto vector_base = vector.begin();
  typename std::vector<T>::size_type down_by = 0;
  for (auto iter = to_remove.cbegin(); 
       iter < to_remove.cend(); 
       iter++, down_by++)
  {
    typename std::vector<T>::size_type next = (iter + 1 == to_remove.cend() 
                                      ? vector.size() 
                                      : *(iter + 1));
    std::move(vector_base + *iter + 1, 
              vector_base + next, 
              vector_base + *iter - down_by);
  }
  vector.resize(vector.size() - to_remove.size());
}

void SemanticFusionInterface::CalculateProjectedProbabilityMap(const std::unique_ptr<ElasticFusionInterface>& map) {
  const int id_width = 640;
  const int id_height = 480;
  const int table_height = num_classes_;
  const int table_width = max_components_;
  renderProbabilityMap(map->GetSurfelIdsGpu(),id_width,id_height,
                       class_probabilities_gpu_->mutable_gpu_data(),
                       table_width,table_height,
                       rendered_class_probabilities_gpu_->mutable_gpu_data());
}

std::shared_ptr<caffe::Blob<float> > SemanticFusionInterface::get_rendered_probability() {
  return rendered_class_probabilities_gpu_;
}

std::shared_ptr<caffe::Blob<float> > SemanticFusionInterface::get_class_max_gpu() {
  return class_max_gpu_;
}

int SemanticFusionInterface::max_num_components() const {
  return max_components_;
}

const float* SemanticFusionInterface::GetClassMaxGpu() {
  return class_max_gpu_->gpu_data();
}

void SemanticFusionInterface::UpdateProbabilityTable(const std::unique_ptr< ElasticFusionInterface >& map) {
  const int new_table_width = map->GetMapSurfelCount();
  const int num_deleted = map->GetMapSurfelDeletedCount();
  const int table_width = max_components_;	// 3000000
  const int table_height = num_classes_;	// 81
  
  updateProbabilityTable(map->GetDeletedSurfelIdsGpu(),num_deleted,current_table_size_,
			    table_width,table_height,new_table_width,
			    class_probabilities_gpu_->gpu_data(),class_probabilities_gpu_buffer_->mutable_gpu_data(),
			    class_max_gpu_->gpu_data(),class_max_gpu_buffer_->mutable_gpu_data());
  
  class_probabilities_gpu_.swap(class_probabilities_gpu_buffer_);
  class_max_gpu_.swap(class_max_gpu_buffer_);
  current_table_size_ = new_table_width;
}

int SemanticFusionInterface::UpdateSurfelProbabilities(const int surfel_id, 
                                                        const std::vector<float>& class_probs) 
{
  assert(static_cast<int>(class_probabilities_.size()) > surfel_id);
  std::vector<float>& surfel_probs = class_probabilities_[surfel_id];
  assert(static_cast<int>(class_probs.size()) == num_classes_);
  assert(static_cast<int>(surfel_probs.size()) == num_classes_);
  float normalisation_denominator = 0.0;
  for (int class_id = 0; class_id < num_classes_; class_id++) {
    surfel_probs[class_id] *= class_probs[class_id];
    normalisation_denominator += surfel_probs[class_id];
  }
  float max_prob = 0.0;
  int max_class = -1;
  for (int class_id = 0; class_id < num_classes_; class_id++) {
    surfel_probs[class_id] /= normalisation_denominator;
    if (surfel_probs[class_id] >= max_prob) {
      max_prob = surfel_probs[class_id];
      max_class = class_id;
    }
  }
  if (max_prob >= colour_threshold_) {
    return max_class;
  }
  return -1;
}

void SemanticFusionInterface::UpdateProbabilities(const Eigen::Matrix4f &mask_pose, cv::Mat& scores_matrix, cv::Mat& class_ids_matrix, const std::unique_ptr< ElasticFusionInterface >& map) {
  float *scores = (float*)scores_matrix.data;
  float *class_ids = (float*)class_ids_matrix.data;
  const int id_width = 640;				// 640
  const int id_height = 480;				// 480
  const int prob_channels = 81;			// 14
  const int map_size = 3000000;	// 3000000
  
  map->elastic_fusion_->getIndexMap().renderSurfelIds(mask_pose, map->elastic_fusion_->getTick(), map->elastic_fusion_->getGlobalModel().model(), map->elastic_fusion_->confidenceThreshold,
                                                      map->elastic_fusion_->maxDepthProcessed, map->elastic_fusion_->timeDelta);
  
  cudaSetDevice(3);
  // Allocate space for device copy of MaskRCNN outpus
  float *scores_gpu, *class_ids_gpu;
  cudaMalloc((void **)&scores_gpu, (sizeof(float)*id_width*id_height));
  cudaMalloc((void **)&class_ids_gpu, (sizeof(float)*id_width*id_height));
  // Copy MaskRCNN outputs to device
  cudaMemcpy(scores_gpu, scores, (sizeof(float)*id_width*id_height), cudaMemcpyHostToDevice);
  cudaMemcpy(class_ids_gpu, class_ids, (sizeof(float)*id_width*id_height), cudaMemcpyHostToDevice);
  
  fuseSemanticProbabilities(map->GetSurfelIdsGpu(), scores_gpu, class_ids_gpu, id_width, id_height, prob_channels, class_probabilities_gpu_->mutable_gpu_data(), class_max_gpu_->mutable_gpu_data(), map_size);
  map->UpdateSurfelClassGpu(map_size,class_max_gpu_->gpu_data(),class_max_gpu_->gpu_data() + map_size,colour_threshold_);
  
  cudaFree(scores_gpu);
  cudaFree(class_ids_gpu);
  
  // reset MaskRCNN output objects
  scores_matrix = cv::Mat(480, 640, CV_32FC1, cv::Scalar(0.0));
  class_ids_matrix = cv::Mat(480, 640, CV_32FC1, cv::Scalar(0.0));
}

void SemanticFusionInterface::CRFUpdate(const std::unique_ptr<ElasticFusionInterface>& map, const int iterations) {
  float* surfel_map = map->GetMapSurfelsGpu();
  // We very inefficiently allocate and clear a chunk of memory for every CRF update
  float * my_surfels = new float[current_table_size_ * 12];
  cudaMemcpy(my_surfels,surfel_map, sizeof(float) * current_table_size_ * 12, cudaMemcpyDeviceToHost);
  // Get the semantic table on CPU and add as unary potentials
  float* prob_table = class_probabilities_gpu_->mutable_cpu_data();

  std::vector<int> valid_ids;
  for (int i = 0; i < current_table_size_; ++i) {
    valid_ids.push_back(i);
  }
  std::vector<float> unary_potentials(valid_ids.size() * num_classes_);
  for(int i = 0; i < static_cast<int>(valid_ids.size()); ++i) {
    int id = valid_ids[i];
    for (int j = 0; j < num_classes_; ++j) {
       unary_potentials[i * num_classes_ + j] = -log(prob_table[j * max_components_ + id] + 1.0e-12);
    }
  }
  DenseCRF3D crf(valid_ids.size(),num_classes_,0.05,20,0.1);
  crf.setUnaryEnergy(unary_potentials.data());
  // Add pairwise energies
  crf.addPairwiseGaussian(my_surfels,3,valid_ids);
  crf.addPairwiseBilateral(my_surfels,10,valid_ids);
  // Finally read the values back to the probability table 
  float* resulting_probs = crf.runInference(iterations, 1.0);
  for (int i = 0; i < static_cast<int>(valid_ids.size()); ++i) {
    for (int j = 0; j < num_classes_; ++j) {
	  const int id = valid_ids[i];
      // Sometimes it returns nan resulting probs... filter these out
      if (resulting_probs[i * num_classes_ + j] > 0.0 && resulting_probs[i * num_classes_ + j] < 1.0) {
        prob_table[j * max_components_ + id] = resulting_probs[i * num_classes_ + j];
      }
    }
  }
  const float* gpu_prob_table = class_probabilities_gpu_->gpu_data();
  float* gpu_max_map = class_max_gpu_->mutable_gpu_data();
  updateMaxClass(current_table_size_,gpu_prob_table,num_classes_,gpu_max_map,max_components_);
  map->UpdateSurfelClassGpu(max_components_,class_max_gpu_->gpu_data(),class_max_gpu_->gpu_data() + max_components_,colour_threshold_);
  delete [] my_surfels;
}

void SemanticFusionInterface::SaveArgMaxPredictions(std::string& filename,const std::unique_ptr<ElasticFusionInterface>& map) {
  const float* max_prob = class_max_gpu_->cpu_data() + max_components_;
  const float* max_class = class_max_gpu_->cpu_data();
  const std::vector<int>& surfel_ids = map->GetSurfelIdsCpu();
  cv::Mat argmax_image(240,320,CV_8UC3);
  for (int h = 0; h < 240; ++h) {
    for (int w = 0; w < 320; ++w) {
      float this_max_prob = 0.0;
      int this_max_class = 0;
      const int start = 0;
      const int end = 2;
      for (int x = start; x < end; ++x) {
        for (int y = start; y < end; ++y) {
          int id = surfel_ids[((h * 2) + y) * 640 + (w * 2 + x)];
          if (id > 0 && id < current_table_size_) {
            if (max_prob[id] > this_max_prob) {
              this_max_prob = max_prob[id];
              this_max_class = max_class[id];
            }
          }
        }
      }
      argmax_image.at<cv::Vec3b>(h,w)[0] = static_cast<int>(this_max_class);
      argmax_image.at<cv::Vec3b>(h,w)[1] =  static_cast<int>(this_max_class);
      argmax_image.at<cv::Vec3b>(h,w)[2] =  static_cast<int>(this_max_class);
    }
  }
  cv::imwrite(filename,argmax_image);
}
