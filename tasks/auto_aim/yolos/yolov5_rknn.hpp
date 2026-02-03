#ifndef AUTO_AIM__YOLOV5_RKNN_HPP
#define AUTO_AIM__YOLOV5_RKNN_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLOV5_RKNN : public YOLOBase
{
public:
  YOLOV5_RKNN(const std::string & config_path, bool debug);
  ~YOLOV5_RKNN();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  const float nms_threshold_ = 0.3F;
  const float score_threshold_ = 0.7F;
  double min_confidence_, binary_threshold_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;
  friend class MultiThreadDetector;

  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);

  bool init_rknn(const std::string & model_path);
  bool infer(const cv::Mat & img_rgb_u8, std::vector<rknn_output> & outputs);
  void release_outputs(std::vector<rknn_output> & outputs);
  static bool read_file(const std::string & path, std::vector<uint8_t> & data);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_RKNN_HPP
