#ifndef AUTO_AIM__YOLOV5_RKNN_HPP
#define AUTO_AIM__YOLOV5_RKNN_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <list>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <string>
#include <thread>
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
  struct RknnContext
  {
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
  };

  struct InferResult
  {
    bool ok = false;
    size_t ctx_index = 0;
    int rows = 0;
    int cols = 0;
    long long infer_us = 0;
    std::vector<float> output;
  };

  struct InferJob
  {
    cv::Mat input_rgb;
    std::promise<InferResult> promise;
  };

  static constexpr size_t kRknnContextCount = 3;

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

  std::array<RknnContext, kRknnContextCount> ctxs_{};
  std::atomic<uint32_t> next_ctx_{0};

  std::array<std::thread, kRknnContextCount> workers_{};
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::deque<InferJob> job_queue_;
  bool stop_workers_ = false;
  bool workers_started_ = false;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);

  bool init_rknn(const std::string & model_path);
  void start_workers();
  void stop_workers();
  void worker_loop(size_t ctx_index);
  bool infer(
    const cv::Mat & img_rgb_u8, std::vector<rknn_output> & outputs, size_t ctx_index);
  void release_outputs(std::vector<rknn_output> & outputs, size_t ctx_index);
  static bool read_file(const std::string & path, std::vector<uint8_t> & data);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_RKNN_HPP
