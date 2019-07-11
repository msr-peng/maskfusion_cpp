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
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>
#include <thread>
#include <chrono>

#include <caffe/caffe.hpp>

#include <cnn_interface/CaffeInterface.h>
#include <map_interface/ElasticFusionInterface.h>
#include <semantic_fusion/SemanticFusionInterface.h>
#include <utilities/LiveLogReader.h>
#include <utilities/RawLogReader.h>
#include <utilities/PNGLogReader.h>
#include <utilities/Types.h>

#include <gui/Gui.h>

#include <../mask_rcnn_pytorch/modelprepare.h>
#include <../mask_rcnn_pytorch/inference.h>

std::vector<ClassColour> load_colour_scheme(std::string filename, int num_classes) {
  std::vector<ClassColour> colour_scheme(num_classes);
  std::ifstream file(filename);
  std::string str;
  int line_number = 1;
  while (std::getline(file, str))
  {
    std::istringstream buffer(str);
    std::string textual_prefix;
    int id, r, g, b;
    if (line_number > 2) {
      buffer >> textual_prefix >> id >> r >> g >> b;
      ClassColour class_colour(textual_prefix,r,g,b);
      assert(id < num_classes);
      colour_scheme[id] = class_colour;
    }
    line_number++;
  }
  return colour_scheme;
}

void infer(cv::Mat &input_image, cv::Mat &output_image, cv::Mat &scores_matrix, cv::Mat &class_ids_matrix, MaskRCNN &model, std::shared_ptr<InferenceConfig> config, bool &IsMaskRCNNInferring) {
  IsMaskRCNNInferring = true;
  inference(input_image, output_image, scores_matrix, class_ids_matrix, model, config);
  IsMaskRCNNInferring = false;
  auto stop = std::chrono::steady_clock::now();
}

int main(int argc, char *argv[]) {
  // Set MaskRCNN Parameters
  std::string model_path = "/home/src/MaskFusion_cpp/model/mask_rcnn_coco.dat";
  cv::Mat input_image_bgr;
  cv::Mat output_image(480, 640, CV_8UC3, cv::Scalar(255,255,255));
  cv::Mat scores_matrix(480, 640, CV_32FC1, cv::Scalar(0.0));
  cv::Mat class_ids_matrix(480, 640, CV_32FC1, cv::Scalar(0.0));;
  bool IsMaskRCNNInferring = false;
  Eigen::Matrix4f mask_pose = Eigen::Matrix4f::Identity();
  
  auto config = std::make_shared<InferenceConfig>();
  std::string model_dir = "./";
  // Create model object.
  MaskRCNN model(model_dir, config);
  
  if (config->gpu_count > 0)
    model->to(torch::DeviceType::CUDA);
  
  std::tie(model, config) = model_prepare(model_path);
  
  caffe::Caffe::SetDevice(1);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // CNN Skip params
  const int cnn_skip_frames = 10;
  // Option CPU-based CRF smoothing
  const bool use_crf = false;
  const int crf_skip_frames = 500;
  const int crf_iterations = 10;
  const int new_num_classes = 81;
  std::cout<<"Network produces "<<new_num_classes<<" output classes"<<std::endl;
  // Check the class colour output and the number of classes matches
  std::vector<ClassColour> class_colour_lookup = load_colour_scheme("../class_colour_scheme.data",new_num_classes);
  std::unique_ptr<SemanticFusionInterface> mask_fusion(new SemanticFusionInterface(new_num_classes,100));
  // Initialise the Gui, Map, and Kinect Log Reader
  const int width = 640;
  const int height = 480;
  Resolution::getInstance(width, height);
  Intrinsics::getInstance(528, 528, 320, 240);
  // Choose the input Reader, live for a running OpenNI device, PNG for textfile lists of PNG frames
  std::unique_ptr<LogReader> log_reader;
  if (argc > 2) {
    // This is official log_reader
    //log_reader.reset(new PNGLogReader(argv[1],argv[2]));
    std::string logFile;
    Parse::get().arg(argc, argv, "-l", logFile);
    if(logFile.length()) {
      log_reader.reset(new RawLogReader(logFile, false));
    }
    else {
    log_reader.reset(new PNGLogReader(argv[1],argv[2]));
    }
  } else {
    log_reader.reset(new LiveLogReader("./live",false));
    if (!log_reader->is_valid()) {
      std::cout<<"Error initialising live device..."<<std::endl;
      return 1;
    }
  }
  // Initialise gui and map after log_reader for existing OpenGL context
  std::unique_ptr<Gui> gui(new Gui(true,class_colour_lookup,640,480));
  std::unique_ptr<ElasticFusionInterface> map(new ElasticFusionInterface());
  if (!map->Init(class_colour_lookup)) {
    std::cout<<"ElasticFusionInterface init failure"<<std::endl;
  }
  // Frame numbers for logs
  int frame_num = 0;
  while(!pangolin::ShouldQuit() && log_reader->hasMore()) {
    gui->preCall();
    // Read and perform an elasticFusion update
    if (!gui->paused() || gui->step()) {
      log_reader->getNext();
      map->setTrackingOnly(gui->tracking());
      if (!map->ProcessFrame(log_reader->rgb, log_reader->depth,log_reader->timestamp)) {
        std::cout<<"Elastic fusion lost!"<<argv[1]<<std::endl;
        return 1;
      }
      // This queries the map interface to update the indexes within the table 
      // It MUST be done everytime ProcessFrame is performed as long as the map
      // is not performing tracking only (i.e. fine to not call, when working
      // with a static map)
      if(!gui->tracking()) {
        mask_fusion->UpdateProbabilityTable(map);
      }
      if (use_crf && frame_num % crf_skip_frames == 0) {
        std::cout<<"Performing CRF Update..."<<std::endl;
        mask_fusion->CRFUpdate(map,crf_iterations);
      }
    }
    frame_num++;
    // This is for outputting the predicted frames
    if (log_reader->isLabeledFrame()) {
      // Change this to save the NYU raw label predictions to a folder.
      // Note these are raw, without the CNN fall-back predictions where there
      // is no surfel to give a prediction.
      std::string save_dir("./");
      std::string label_dir(log_reader->getLabelFrameId());
      std::string suffix("_label.png");
      save_dir += label_dir;
      save_dir += suffix;
      std::cout<<"Saving labeled frame to "<<save_dir<<std::endl;
      mask_fusion->SaveArgMaxPredictions(save_dir,map);
    }
    gui->renderMap(map);
    // This one requires the size of the segmentation display to be set in the Gui constructor to 224,224
    gui->displayImg("raw",map->getRawImageTexture());
    
    if(!IsMaskRCNNInferring) {
      mask_fusion->UpdateProbabilities(mask_pose, scores_matrix, class_ids_matrix, map);
      mask_pose = map->elastic_fusion_->getCurrPose();
      cv::Size image_size(640, 480);
      cv::Mat input_image_rgb(image_size, CV_8UC3, log_reader->rgb);
      cv::cvtColor(input_image_rgb, input_image_bgr, cv::COLOR_RGB2BGR);
      std::thread MaskRCNNThread(infer, std::ref(input_image_bgr), std::ref(output_image), std::ref(scores_matrix), std::ref(class_ids_matrix), std::ref(model), std::ref(config), std::ref(IsMaskRCNNInferring));
      MaskRCNNThread.detach();
    }
    
    pangolin::GlTexture imageTexture(640,480,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
    imageTexture.Upload(output_image.data,GL_BGR,GL_UNSIGNED_BYTE);

    gui->displayMaskRCNNInference("infer",imageTexture);
    // This is to display a predicted semantic segmentation from the fused map
    mask_fusion->CalculateProjectedProbabilityMap(map);
    gui->displayArgMaxClassColouring("segmentation",mask_fusion->get_rendered_probability()->mutable_gpu_data(),
                                     new_num_classes,mask_fusion->get_class_max_gpu()->gpu_data(),
                                     mask_fusion->max_num_components(),map->GetSurfelIdsGpu(),0.0);
    gui->postCall();
    
    if (gui->reset()) {
      map.reset(new ElasticFusionInterface());
      if (!map->Init(class_colour_lookup)) {
        std::cout<<"ElasticFusionInterface init failure"<<std::endl;
      }
    }
    if (gui->save()) {
      map->elastic_fusion_->savePly();
      std::cout << "Saved surfel map!" << std::endl;
      float *class_ids_table = new float[mask_fusion->max_num_components()]();
      cudaMemcpy(class_ids_table, mask_fusion->GetClassMaxGpu(), sizeof(float)*mask_fusion->max_num_components(), cudaMemcpyDeviceToHost);
      map->elastic_fusion_->savePly(class_ids_table, map->GetClassColorLookup());
      std::cout << "Saved semantic surfel map!" << std::endl;
      delete [] class_ids_table;
    }
  }
  std::cout<<"Finished MaskFusion"<<std::endl;
  return 0;
}
