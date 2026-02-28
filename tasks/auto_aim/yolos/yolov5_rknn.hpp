#ifndef AUTO_AIM__YOLOV5_RKNN_HPP
#define AUTO_AIM__YOLOV5_RKNN_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <list>
#include <mutex>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLOV5_RKNN : public YOLOBase
{
public:
  using JobId = uint64_t;
  static constexpr JobId kInvalidJobId = 0;

  YOLOV5_RKNN(const std::string & config_path, bool debug);
  ~YOLOV5_RKNN();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  JobId submit(const cv::Mat & bgr_img, int frame_count);
  std::list<Armor> wait(JobId job_id);
  bool try_wait(JobId job_id, std::list<Armor> & armors);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  struct RknnContext
  {
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;

    // Zero-copy IO memory
    bool use_io_mem = false;
    rknn_tensor_attr io_input_attr{};
    rknn_tensor_mem * input_mem = nullptr;
    std::vector<rknn_tensor_attr> io_output_attrs;
    std::vector<rknn_tensor_mem *> output_mems;
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

  struct PendingJob
  {
    double scale = 1.0;
    cv::Mat raw_img;
    int frame_count = -1;
    std::future<InferResult> future;
    std::chrono::steady_clock::time_point t_begin;
    std::chrono::steady_clock::time_point t_pre_end;
    std::chrono::steady_clock::time_point t_submit;
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

  std::mutex pending_mu_;
  std::unordered_map<JobId, PendingJob> pending_jobs_;
  std::atomic<JobId> next_job_id_{1};
  
  std::mutex postprocess_mu_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor, const cv::Mat & img) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  // Removed: unused member `double sigmoid(double x)` — use anonymous-namespace SigmoidFast instead

  bool preprocess(const cv::Mat & raw_img, cv::Mat & input_rgb, double & scale) const;
  std::future<InferResult> enqueue_infer(const cv::Mat & input_rgb);

  bool init_rknn(const std::string & model_path);
  void start_workers();
  void stop_workers();
  void worker_loop(size_t ctx_index);
  bool infer(const cv::Mat & img_rgb_u8, size_t ctx_index);
  void release_outputs(size_t ctx_index);
  bool init_io_mem(RknnContext & ctx_item);
  void destroy_io_mem(RknnContext & ctx_item);
  InferResult decode_outputs(size_t ctx_index);
  void log_timing(
    int frame_count, size_t ctx_index, double pre_ms, double wait_ms,
    double infer_ms, double post_ms, double total_ms, int rows, int cols) const;
  static bool read_file(const std::string & path, std::vector<uint8_t> & data);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_RKNN_HPP
