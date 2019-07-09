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

#include <iostream>
#include <vector>
#include <stdio.h>
#include <assert.h> 

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
        abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } 
}

__global__ 
void semanticTableUpdate(cudaTextureObject_t ids, const float *scores, const float *class_ids,
                             const int ids_width, const int ids_height, const int prob_channels,
			     float* map_table, float* map_max, const int map_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int surfel_id = tex2D<int>(ids,x,y);
    // New uniqueness code
    const int check_patch = 16;
    const int x_min = (x - check_patch) < 0 ? 0 : (x - check_patch);
    const int x_max = (x + check_patch) > 640 ? 640 : (x + check_patch);
    const int y_min = (y - check_patch) < 0 ? 0 : (y - check_patch);
    const int y_max = (y + check_patch) > 480 ? 480 : (y + check_patch);
    int first_h, first_w;
    for (int h = y_min; h < y_max; ++h) {
        for (int w = x_min; w < x_max; ++w) {
            int other_surfel_id = tex2D<int>(ids,w,h);
            if (other_surfel_id == surfel_id) {
                first_h = h;
                first_w = w;
                break;
            }
        }
    }
    if (first_h != y || first_w != x) {
        surfel_id = 0;
    }
    if (surfel_id > 0) {
        const float* probability = scores + (y * ids_width + x);
        float* prior_probability = map_table + surfel_id;
        float total = 0.0;
	
        for (unsigned char class_id = 0; class_id < prob_channels; ++class_id) {
          // if pixel (x, y) is not inferred as background by MaskRCNN
          if (fabsf(class_ids[y * ids_width + x]) >= 1e-5) {
	    if (fabsf(class_ids[y * ids_width + x] - class_id) <= 1e-5) {
	      prior_probability[0] *= probability[0];
	    }
	    else {
	      prior_probability[0] *= ((1.0f - probability[0])/(prob_channels - 1));
	    }
	  }
	  else {
	    prior_probability[0] *= (1.0f / prob_channels);
	  }
            total += prior_probability[0];
            prior_probability += map_size;
        }
        // Reset the pointers to the beginning again
        prior_probability = map_table + surfel_id;
        float max_probability = 0.0;
        int max_class = -1;
        float new_total = 0.0;
        for (int class_id = 0; class_id < prob_channels; ++class_id) {
            // Something has gone unexpectedly wrong - reinitialse
            if (total <= 1e-5) {
                prior_probability[0] = 1.0f / prob_channels;
            } else {
                prior_probability[0] /= total;
                if (prior_probability[0] > max_probability) {
                    max_probability = prior_probability[0];
                    max_class = class_id;
                }
            }
            new_total += prior_probability[0];
            prior_probability += map_size;
        }
        map_max[surfel_id] = static_cast<float>(max_class);
        map_max[surfel_id + map_size] = max_probability;
        map_max[surfel_id + map_size + map_size] += 1.0;
    }
}

__host__ 
void fuseSemanticProbabilities(cudaTextureObject_t ids, const float *scores, const float *class_ids,
                             const int ids_width, const int ids_height, const int prob_channels,
			     float* map_table, float* map_max, const int map_size) {
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock(640/blocks,480/blocks);
    semanticTableUpdate<<<dimGrid,dimBlock>>>(ids,scores,class_ids,ids_width,ids_height,prob_channels,map_table,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__
void updateTable(int n, const int* deleted_ids, const int num_deleted, const int current_table_size,
                 const int prob_width, const int prob_height, const int new_prob_width,
		 float const* probability_table, float* new_probability_table, float const * map_table, float* new_map_table) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const int class_id = index / new_prob_width;
        const int component_id = index - (class_id * new_prob_width);
        const int new_id = (class_id * prob_width) + component_id;
        if (component_id >= num_deleted) {
            // Initialise to prior (prob height is the number of classes)
            new_probability_table[new_id] = 1.0f / prob_height;
            // Reset the max class surfel colouring lookup
            new_map_table[component_id] = -1.0;
            new_map_table[component_id + prob_width] = -1.0;
            new_map_table[component_id + prob_width + prob_width] = 0.0;
        } else {
            int offset = deleted_ids[component_id];
            new_probability_table[new_id] = probability_table[(class_id * prob_width) + offset];
            // Also must update our max class mapping
            new_map_table[component_id] = map_table[offset];
            new_map_table[component_id + prob_width] = map_table[prob_width + offset];
            new_map_table[component_id + prob_width + prob_width] = map_table[prob_width + prob_width + offset];
        }
    }
}

__host__
void updateProbabilityTable(int* filtered_ids, const int num_filtered, const int current_table_size,
                               const int prob_width, const int prob_height, const int new_prob_width,
			       float const* probability_table, float* new_probability_table,
                               float const* map_table, float* new_map_table) {
    const int threads = 512;
    const int num_to_update = new_prob_width * prob_height;
    const int blocks = (num_to_update + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateTable<<<dimGrid,dimBlock>>>(num_to_update,filtered_ids,num_filtered,current_table_size,prob_width,prob_height,new_prob_width,probability_table,new_probability_table,map_table,new_map_table);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void renderProbabilityMapKernel(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int surfel_id = tex2D<int>(ids,x,y);
    int projected_probability_offset = y * ids_width + x;
    int probability_table_offset = surfel_id;
    for (int class_id = 0; class_id < prob_height; ++class_id) {
        if (surfel_id > 0) {
            rendered_probabilities[projected_probability_offset] = probability_table[probability_table_offset];
        } else {
            rendered_probabilities[projected_probability_offset] = ((class_id == 0) ? 1.0 : 0.0);
        }
        projected_probability_offset += (ids_width * ids_height);
        probability_table_offset += prob_width;
    }
}

__host__
void renderProbabilityMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock(ids_width/blocks,ids_height/blocks);
    renderProbabilityMapKernel<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,probability_table,prob_width,prob_height,rendered_probabilities);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void updateMaxClassKernel(const int n, const float* probabilities, const int classes,
                          float* map_max, const int map_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Reset the pointers to the beginning again
        const float* probability = probabilities + index;
        probability += map_size;
        float max_probability = 0.0;
        int max_class = -1;
        for (int class_id = 1; class_id < classes; ++class_id) {
            if (probability[0] > max_probability) {
                max_probability = probability[0];
                max_class = class_id;
            }
            probability += map_size;
        }
        map_max[index] = static_cast<float>(max_class);
        map_max[index + map_size] = max_probability;
    }
}

__host__ 
void updateMaxClass(const int n, const float* probabilities, const int classes,
                    float* map_max, const int map_size)
{
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateMaxClassKernel<<<dimGrid,dimBlock>>>(n,probabilities,classes,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__
void showSurfelIdKernel(cudaTextureObject_t ids) {
  int surfel_id = tex2D<int>(ids,320,240);
  if (tex2D<int>(ids,320,240)) {
    printf("********************** x = 320 && y = 240 : %d \n", surfel_id);
  }
}

__host__
void showSurfelId(cudaTextureObject_t ids) {
  dim3 dimGrid(1);
  dim3 dimBlock(1);
  showSurfelIdKernel<<<dimGrid, dimBlock>>>(ids);
  gpuErrChk(cudaGetLastError());
  gpuErrChk(cudaDeviceSynchronize());
}