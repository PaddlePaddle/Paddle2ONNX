/*#pragma once

#include ""

class PaddleInferenceModel {
  public:
    bool Init(const string &model_dir, const string &cfg_file, Engine_config &engine_config);
  
    bool Predict(const std::vector<cv::Mat> &imgs, std::vector<DetResult> *results);
    
    bool Predict(const std::vector<cv::Mat> &imgs, std::vector<ClsResult> *results);

    bool Predict(const std::vector<cv::Mat> &imgs, std::vector<SegResult> *results);

    bool Predict(const std::vector<cv::Mat> &imgs, std::vector<OcrResult> *results);

  private:
    YAML::Node config_;
    
}*/
