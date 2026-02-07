#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rknn_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace {

constexpr int kImgSize = 640;
constexpr float kConfThresh = 0.65f;
constexpr float kNmsThresh = 0.45f;

// Output layout (match onnx2rknn.py and OpenvinoInfer.cpp)
constexpr int kNumKeypoints = 4; // 4 points
constexpr int kKptStart = 0;     // 0..7 are keypoints (x,y)*4
constexpr int kConfIndex = 8;    // needs sigmoid
constexpr int kColorStart = 9;   // 9..12 (4)
constexpr int kColorEnd = 13;
constexpr int kNumStart = 13; // 13..21 (9)
constexpr int kNumEnd = 22;

float Sigmoid(float x) {
  x = std::max(-50.0f, std::min(50.0f, x));
  return 1.0f / (1.0f + std::exp(-x));
}

float Logit(float p) {
  // avoid inf/NaN
  constexpr float kEps = 1e-6f;
  p = std::max(kEps, std::min(1.0f - kEps, p));
  return std::log(p / (1.0f - p));
}

struct LatStats {
  double avg_ms = 0.0;
  double max_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
};

LatStats ComputeLatStats(std::vector<long long> samples_us) {
  LatStats stats;
  if (samples_us.empty()) {
    return stats;
  }
  long long sum = 0;
  long long max_v = samples_us[0];
  for (long long v : samples_us) {
    sum += v;
    if (v > max_v) {
      max_v = v;
    }
  }
  stats.avg_ms = (sum / 1000.0) / static_cast<double>(samples_us.size());
  stats.max_ms = max_v / 1000.0;

  std::sort(samples_us.begin(), samples_us.end());
  const auto pick = [&](double p) -> double {
    if (samples_us.empty()) {
      return 0.0;
    }
    const double pos = p * static_cast<double>(samples_us.size());
    size_t idx = 0;
    if (pos <= 1.0) {
      idx = 0;
    } else {
      idx = static_cast<size_t>(std::ceil(pos)) - 1;
      if (idx >= samples_us.size()) {
        idx = samples_us.size() - 1;
      }
    }
    return samples_us[idx] / 1000.0;
  };

  stats.p90_ms = pick(0.90);
  stats.p99_ms = pick(0.99);
  return stats;
}



bool ReadFile(const std::string &path, std::vector<uint8_t> &data) {
  FILE *fp = fopen(path.c_str(), "rb");
  if (!fp) {
    return false;
  }
  fseek(fp, 0, SEEK_END);
  long len = ftell(fp);
  if (len <= 0) {
    fclose(fp);
    return false;
  }
  fseek(fp, 0, SEEK_SET);
  data.resize(static_cast<size_t>(len));
  size_t rd = fread(data.data(), 1, data.size(), fp);
  fclose(fp);
  return rd == data.size();
}

struct DetObj {
  float prob = 0.0f; // sigmoid(conf)
  int color_id = -1; // argmax color scores
  int label = -1;    // argmax num scores
  float score_color = 0.0f;
  float score_num = 0.0f; // used for NMS score like OpenvinoInfer.cpp
  cv::Rect rect;
  std::array<cv::Point2f, kNumKeypoints> kpts; // 4 points
};

float IoU_xyxy(const cv::Rect2f &a, const cv::Rect2f &b) {
  float xx1 = std::max(a.x, b.x);
  float yy1 = std::max(a.y, b.y);
  float xx2 = std::min(a.x + a.width, b.x + b.width);
  float yy2 = std::min(a.y + a.height, b.y + b.height);
  float w = std::max(0.0f, xx2 - xx1);
  float h = std::max(0.0f, yy2 - yy1);
  float inter = w * h;
  float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
  float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
  return inter / (area_a + area_b - inter + 1e-9f);
}

void PostprocessSingleHead(const float *out, int rows, int cols,
                           int detect_color, float conf_thresh,
                           float nms_thresh, std::vector<DetObj> &out_objs) {
  out_objs.clear();
  if (!out || rows <= 0 || cols <= 0) {
    return;
  }
  if (cols < kNumEnd) {
    throw std::runtime_error("output cols < 22, unexpected model output");
  }

  const float conf_logit_thresh = Logit(conf_thresh);

  std::vector<cv::Rect> boxes;
  std::vector<float> nms_scores;
  std::vector<DetObj> candidates;
  candidates.reserve(static_cast<size_t>(rows));

  boxes.reserve(static_cast<size_t>(rows));
  nms_scores.reserve(static_cast<size_t>(rows));

  for (int i = 0; i < rows; ++i) {
    const float *p = out + i * cols;

    // Fast prefilter: compare logits instead of running exp() on every row.
    const float conf_logit = p[kConfIndex];
    if (!(conf_logit >= conf_logit_thresh)) {
      continue;
    }
    const float prob = Sigmoid(conf_logit);

    // color argmax
    int color_id = 0;
    float best_color = p[kColorStart];
    for (int j = kColorStart + 1; j < kColorEnd; ++j) {
      if (p[j] > best_color) {
        best_color = p[j];
        color_id = j - kColorStart;
      }
    }

    // num/digit argmax
    int label = 0;
    float best_num = p[kNumStart];
    for (int j = kNumStart + 1; j < kNumEnd; ++j) {
      if (p[j] > best_num) {
        best_num = p[j];
        label = j - kNumStart;
      }
    }

    // Match OpenvinoInfer.cpp filter semantics:
    // drop None/Purple => ids 2,3
    if (color_id == 2 || color_id == 3) {
      continue;
    }
    // detect_color: -1 keep all, 0 blue-only, 1 red-only
    if (detect_color == 0 && color_id == 1) {
      continue;
    }
    if (detect_color == 1 && color_id == 0) {
      continue;
    }

    DetObj obj;
    obj.prob = prob;
    obj.color_id = color_id;
    obj.label = label;
    obj.score_color = best_color;
    obj.score_num = best_num;

    // Read keypoints and compute bbox in one pass.
    float x0 = p[kKptStart + 0];
    float y0 = p[kKptStart + 1];
    obj.kpts[0] = cv::Point2f(x0, y0);
    float x_min = x0, x_max = x0;
    float y_min = y0, y_max = y0;
    for (int k = 1; k < kNumKeypoints; ++k) {
      const float x = p[kKptStart + k * 2 + 0];
      const float y = p[kKptStart + k * 2 + 1];
      obj.kpts[k] = cv::Point2f(x, y);
      x_min = std::min(x_min, x);
      x_max = std::max(x_max, x);
      y_min = std::min(y_min, y);
      y_max = std::max(y_max, y);
    }

    // Clip to input canvas (0..639)
    x_min = std::max(0.0f, std::min(x_min, static_cast<float>(kImgSize - 1)));
    y_min = std::max(0.0f, std::min(y_min, static_cast<float>(kImgSize - 1)));
    x_max = std::max(0.0f, std::min(x_max, static_cast<float>(kImgSize - 1)));
    y_max = std::max(0.0f, std::min(y_max, static_cast<float>(kImgSize - 1)));

    int left = cvFloor(x_min);
    int top = cvFloor(y_min);
    int right = cvCeil(x_max);
    int bottom = cvCeil(y_max);
    // ensure non-empty rect for stable NMS
    right = std::max(right, left + 1);
    bottom = std::max(bottom, top + 1);
    int w = right - left;
    int h = bottom - top;

    obj.rect = cv::Rect(left, top, w, h);

    candidates.push_back(obj);
    boxes.push_back(obj.rect);
    nms_scores.push_back(obj.score_num);
  }

  if (candidates.empty()) {
    return;
  }

  // Match OpenvinoInfer.cpp: NMSBoxes(boxes, confidences=score_num,
  // score_threshold=conf_thresh, nms_threshold=nms_thresh)
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, nms_scores, conf_thresh, nms_thresh, indices);

  out_objs.reserve(indices.size());
  for (int idx : indices) {
    if (idx >= 0 && idx < static_cast<int>(candidates.size())) {
      out_objs.push_back(candidates[static_cast<size_t>(idx)]);
    }
  }
}

class RknnRunner {
public:
  ~RknnRunner() {
    DestroyIOMem();
    if (ctx_ != 0) {
      rknn_destroy(ctx_);
      ctx_ = 0;
    }
  }

  bool Init(const std::string &model_path,
            rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2,
            bool use_io_mem = true) {
    std::vector<uint8_t> model_data;
    if (!ReadFile(model_path, model_data)) {
      std::cerr << "Failed to read model: " << model_path << "\n";
      return false;
    }

    int ret =
        rknn_init(&ctx_, model_data.data(), model_data.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_init failed, ret=" << ret << "\n";
      ctx_ = 0;
      return false;
    }

    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_query IN_OUT_NUM failed, ret=" << ret << "\n";
      return false;
    }
    ret = rknn_set_core_mask (ctx_, core_mask) ;
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_set_core_mask failed, ret=" << ret << "\n";
      return false;
    }

    input_attrs_.resize(io_num_.n_input);
    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
      input_attrs_[i].index = i;
      rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i],
                 sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      output_attrs_[i].index = i;
      rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i],
                 sizeof(rknn_tensor_attr));
    }

    use_io_mem_ = use_io_mem;
    if (use_io_mem_) {
      if (!InitIOMem()) {
        std::cerr << "InitIOMem failed\n";
        return false;
      }
    }

    return true;
  }

  // Returns output[0] as float pointer (valid until outputs_release)
  bool Infer(const cv::Mat &img_rgb_u8, std::vector<rknn_output> &outputs) {
    if (ctx_ == 0) {
      std::cerr << "RKNN context is null\n";
      return false;
    }
    if (img_rgb_u8.empty() || img_rgb_u8.type() != CV_8UC3) {
      std::cerr << "Expect RGB uint8 HWC CV_8UC3\n";
      return false;
    }

    if (use_io_mem_) {
      if (!CopyToInputMem(img_rgb_u8)) {
        return false;
      }
      const int ret = rknn_run(ctx_, nullptr);
      if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed, ret=" << ret << "\n";
        return false;
      }
      outputs.clear();
      return true;
    }

    rknn_input in;
    std::memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = static_cast<uint32_t>(img_rgb_u8.total() * img_rgb_u8.elemSize());
    in.buf = const_cast<unsigned char *>(img_rgb_u8.ptr<unsigned char>());

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, &in);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_inputs_set failed, ret=" << ret << "\n";
      return false;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_run failed, ret=" << ret << "\n";
      return false;
    }

    outputs.assign(io_num_.n_output, {});
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_outputs_get failed, ret=" << ret << "\n";
      outputs.clear();
      return false;
    }
    // rknn_perf_detail perf_detail;
    // ret = rknn_query(ctx_,RKNN_QUERY_PERF_DETAIL,&perf_detail,sizeof(perf_detail));
    // printf("-->模型逐层耗时:%s\n",perf_detail.perf_data);

    return true;
  }

  void ReleaseOutputs(std::vector<rknn_output> &outputs) {
    if (use_io_mem_) {
      outputs.clear();
      return;
    }
    if (ctx_ == 0 || outputs.empty()) {
      return;
    }
    rknn_outputs_release(ctx_, outputs.size(), outputs.data());
    outputs.clear();
  }

  const float *Output0FloatPtr() const {
    if (!use_io_mem_ || output_mems_.empty() || !output_mems_[0]) {
      return nullptr;
    }
    return reinterpret_cast<const float *>(output_mems_[0]->virt_addr);
  }

  size_t Output0Elems() const {
    if (output_attrs_.empty()) {
      return 0;
    }
    return static_cast<size_t>(output_attrs_[0].n_elems);
  }

  const rknn_tensor_attr &OutputAttr(size_t idx) const {
    return output_attrs_.at(idx);
  }

private:
  bool InitIOMem() {
    DestroyIOMem();

    if (io_num_.n_input < 1 || io_num_.n_output < 1) {
      std::cerr << "Unexpected io num\n";
      return false;
    }

    io_input_attr_ = input_attrs_[0];
    io_input_attr_.type = RKNN_TENSOR_UINT8;
    io_input_attr_.fmt = RKNN_TENSOR_NHWC;

    input_mem_ = rknn_create_mem(ctx_, io_input_attr_.size_with_stride);
    if (!input_mem_) {
      std::cerr << "rknn_create_mem(input) failed\n";
      return false;
    }

    int ret = rknn_set_io_mem(ctx_, input_mem_, &io_input_attr_);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_set_io_mem(input) failed, ret=" << ret << "\n";
      return false;
    }

    output_mems_.resize(io_num_.n_output, nullptr);
    io_output_attrs_ = output_attrs_;
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      io_output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
      const int out_bytes = io_output_attrs_[i].n_elems * static_cast<int>(sizeof(float));
      output_mems_[i] = rknn_create_mem(ctx_, out_bytes);
      if (!output_mems_[i]) {
        std::cerr << "rknn_create_mem(output[" << i << "]) failed\n";
        return false;
      }
      ret = rknn_set_io_mem(ctx_, output_mems_[i], &io_output_attrs_[i]);
      if (ret != RKNN_SUCC) {
        std::cerr << "rknn_set_io_mem(output[" << i << "]) failed, ret=" << ret << "\n";
        return false;
      }
    }

    return true;
  }

  void DestroyIOMem() {
    if (ctx_ == 0) {
      return;
    }
    if (input_mem_) {
      rknn_destroy_mem(ctx_, input_mem_);
      input_mem_ = nullptr;
    }
    for (auto &m : output_mems_) {
      if (m) {
        rknn_destroy_mem(ctx_, m);
        m = nullptr;
      }
    }
    output_mems_.clear();
    io_output_attrs_.clear();
  }

  bool CopyToInputMem(const cv::Mat &img_rgb_u8) {
    if (!input_mem_ || !input_mem_->virt_addr) {
      std::cerr << "input_mem is null\n";
      return false;
    }

    int req_h = 0;
    int req_w = 0;
    int req_c = 0;
    if (io_input_attr_.n_dims >= 4) {
      req_h = io_input_attr_.dims[1];
      req_w = io_input_attr_.dims[2];
      req_c = io_input_attr_.dims[3];
    } else {
      std::cerr << "Unexpected input dims\n";
      return false;
    }

    if (img_rgb_u8.rows != req_h || img_rgb_u8.cols != req_w || img_rgb_u8.channels() != req_c) {
      std::cerr << "Input image shape mismatch, expect "
                << req_w << "x" << req_h << "x" << req_c << "\n";
      return false;
    }

    const int width = req_w;
    const int height = req_h;
    const int channel = req_c;
    const int stride = io_input_attr_.w_stride > 0 ? io_input_attr_.w_stride : width;

    uint8_t *dst = reinterpret_cast<uint8_t *>(input_mem_->virt_addr);
    const uint8_t *src = img_rgb_u8.ptr<uint8_t>();

    if (stride == width) {
      std::memcpy(dst, src, static_cast<size_t>(width) * height * channel);
    } else {
      const int src_wc = width * channel;
      const int dst_wc = stride * channel;
      for (int h = 0; h < height; ++h) {
        std::memcpy(dst + static_cast<size_t>(h) * dst_wc,
                    src + static_cast<size_t>(h) * src_wc,
                    static_cast<size_t>(src_wc));
      }
    }
    return true;
  }

  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  bool use_io_mem_ = false;
  rknn_tensor_attr io_input_attr_{};
  std::vector<rknn_tensor_attr> io_output_attrs_;
  rknn_tensor_mem *input_mem_ = nullptr;
  std::vector<rknn_tensor_mem *> output_mems_;
};

struct BenchResult {
  int runs = 0;
  double wall_ms = 0.0;
  double avg_infer_ms = 0.0;
  double avg_post_ms = 0.0;
  double avg_total_ms = 0.0;
  size_t last_det_count = 0;
  LatStats infer_stats;
  LatStats post_stats;
  LatStats total_stats;
};

struct JobResult {
  long long infer_us = 0;
  long long post_us = 0;
  size_t det_count = 0;
  int ok = 0;
};

bool PinThisThreadToCpu(int cpu_id) {
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::cerr << "pthread_setaffinity_np failed rc=" << rc
              << " cpu=" << cpu_id << "\n";
    return false;
  }
  return true;
#else
  (void)cpu_id;
  return false;
#endif
}

BenchResult RunSequential(RknnRunner &runner, const cv::Mat &img_rgb, int rows,
                          int cols, int detect_color, int runs) {
  BenchResult r;
  r.runs = runs;

  double infer_ms_sum = 0.0;
  double post_ms_sum = 0.0;
  size_t last_det_count = 0;

  std::vector<long long> infer_us_list;
  std::vector<long long> post_us_list;
  std::vector<long long> total_us_list;
  infer_us_list.reserve(static_cast<size_t>(runs));
  post_us_list.reserve(static_cast<size_t>(runs));
  total_us_list.reserve(static_cast<size_t>(runs));

  std::vector<rknn_output> outputs;
  std::vector<DetObj> dets;

  const auto wall_t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!runner.Infer(img_rgb, outputs)) {
      runner.ReleaseOutputs(outputs);
      throw std::runtime_error("Infer failed");
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    const float *out0 = runner.Output0FloatPtr();
    if (!out0) {
      out0 = reinterpret_cast<const float *>(outputs.empty() ? nullptr : outputs[0].buf);
    }
    if (!out0) {
      runner.ReleaseOutputs(outputs);
      throw std::runtime_error("Output buffer is null");
    }

    PostprocessSingleHead(out0, rows, cols, detect_color, kConfThresh,
                          kNmsThresh, dets);
    const auto t2 = std::chrono::high_resolution_clock::now();

    infer_ms_sum +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    post_ms_sum +=
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const long long infer_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const long long post_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    infer_us_list.push_back(infer_us);
    post_us_list.push_back(post_us);
    total_us_list.push_back(infer_us + post_us);
    last_det_count = dets.size();

    runner.ReleaseOutputs(outputs);
  }
  const auto wall_t1 = std::chrono::high_resolution_clock::now();

  r.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0)
                  .count();
  r.avg_infer_ms = infer_ms_sum / runs;
  r.avg_post_ms = post_ms_sum / runs;
  r.avg_total_ms = (infer_ms_sum + post_ms_sum) / runs;
  r.last_det_count = last_det_count;
  r.infer_stats = ComputeLatStats(std::move(infer_us_list));
  r.post_stats = ComputeLatStats(std::move(post_us_list));
  r.total_stats = ComputeLatStats(std::move(total_us_list));
  return r;
}

class JobQueue {
public:
  void Push(int job_id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push(job_id);
    }
    cv_.notify_one();
  }

  bool Pop(int &job_id) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
    if (q_.empty()) {
      return false;
    }
    job_id = q_.front();
    q_.pop();
    return true;
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<int> q_;
  bool stop_ = false;
};

template <typename T>
class BoundedQueue {
public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
      capacity_ = 1;
    }
  }

  bool Push(T item) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_not_full_.wait(lk, [&] { return closed_ || q_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    q_.push(std::move(item));
    cv_not_empty_.notify_one();
    return true;
  }

  bool Pop(T &item) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) {
      return false;
    }
    item = std::move(q_.front());
    q_.pop();
    cv_not_full_.notify_one();
    return true;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      closed_ = true;
    }
    cv_not_full_.notify_all();
    cv_not_empty_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_not_full_;
  std::condition_variable cv_not_empty_;
  std::queue<T> q_;
  size_t capacity_ = 1;
  bool closed_ = false;
};

struct PostItem {
  int job_id = -1;
  long long infer_us = 0;
  int ok = 0;
  std::vector<float> out; // copy of output[0] as float
};

class PostQueue {
public:
  void Push(PostItem item) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push(std::move(item));
    }
    cv_.notify_one();
  }

  bool Pop(PostItem &item) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
    if (q_.empty()) {
      return false;
    }
    item = std::move(q_.front());
    q_.pop();
    return true;
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<PostItem> q_;
  bool stop_ = false;
};

BenchResult RunPipeline3Submit2PostPinned(const std::string &model_path,
                                         const cv::Mat &img_rgb, int rows,
                                         int cols, int detect_color, int runs,
                                         bool print_ordered_per_run) {
  BenchResult r;
  r.runs = runs;

  constexpr int kSubmitThreads = 3;
  constexpr int kNpuWorkers = 3;
  constexpr int kPostThreads = 1;
  constexpr size_t kInflightDepth = 8; // 6~8

  // Submit threads bound to CPU1/2/3, post threads bound to CPU7.
  const std::array<int, kSubmitThreads> submit_cpus = {1, 2, 3};
  const std::array<int, kPostThreads> post_cpus = {7};
  const std::array<int, kNpuWorkers> npu_cpus = {4, 5, 6};

  // NPU contexts bound to NPU0/1/2.
  const std::array<rknn_core_mask, kNpuWorkers> npu_masks = {
      RKNN_NPU_CORE_0,
      RKNN_NPU_CORE_1,
      RKNN_NPU_CORE_2,
  };

  std::array<RknnRunner, kNpuWorkers> runners;
  for (int i = 0; i < kNpuWorkers; ++i) {
    if (!runners[static_cast<size_t>(i)].Init(model_path,
                                              npu_masks[static_cast<size_t>(i)])) {
      throw std::runtime_error("Init failed for one of NPU cores");
    }
  }

  BoundedQueue<int> inflight_q(kInflightDepth);
  PostQueue completion_q;
  std::vector<JobResult> results(static_cast<size_t>(runs));

  std::atomic<int> next_job{0};

  // Post threads
  std::array<std::thread, kPostThreads> post_threads;
  for (int p = 0; p < kPostThreads; ++p) {
    post_threads[static_cast<size_t>(p)] = std::thread([&, p]() {
      PinThisThreadToCpu(post_cpus[static_cast<size_t>(p)]);
      std::vector<DetObj> dets;
      PostItem item;
      while (completion_q.Pop(item)) {
        JobResult jr;
        jr.infer_us = item.infer_us;
        jr.ok = item.ok;

        long long post_us = 0;
        size_t det_count = 0;
        if (item.ok && !item.out.empty()) {
          try {
            const auto t1 = std::chrono::high_resolution_clock::now();
            PostprocessSingleHead(item.out.data(), rows, cols, detect_color,
                                  kConfThresh, kNmsThresh, dets);
            const auto t2 = std::chrono::high_resolution_clock::now();
            post_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                    .count();
            det_count = dets.size();
          } catch (const std::exception &e) {
            std::cerr << "Postprocess error: " << e.what() << "\n";
            jr.ok = 0;
          }
        }

        jr.post_us = post_us;
        jr.det_count = det_count;
        if (item.job_id >= 0 && item.job_id < runs) {
          results[static_cast<size_t>(item.job_id)] = jr;
        }
      }
    });
  }

  // NPU worker threads: pop from in-flight queue, run infer, push to completion.
  std::array<std::thread, kNpuWorkers> npu_threads;
  for (int w = 0; w < kNpuWorkers; ++w) {
    npu_threads[static_cast<size_t>(w)] = std::thread([&, w]() {
      PinThisThreadToCpu(npu_cpus[static_cast<size_t>(w)]);
      std::vector<rknn_output> outputs;
      int job_id = -1;
      while (inflight_q.Pop(job_id)) {
        PostItem item;
        item.job_id = job_id;

        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!runners[static_cast<size_t>(w)].Infer(img_rgb, outputs)) {
          runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
          const auto t_fail = std::chrono::high_resolution_clock::now();
          item.infer_us =
              std::chrono::duration_cast<std::chrono::microseconds>(t_fail - t0)
                  .count();
          item.ok = 0;
          completion_q.Push(std::move(item));
          continue;
        }

        const float *out0 = runners[static_cast<size_t>(w)].Output0FloatPtr();
        if (!out0) {
          out0 = reinterpret_cast<const float *>(outputs.empty() ? nullptr : outputs[0].buf);
        }
        if (!out0) {
          runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
          const auto t_null = std::chrono::high_resolution_clock::now();
          item.infer_us =
              std::chrono::duration_cast<std::chrono::microseconds>(t_null - t0)
                  .count();
          item.ok = 0;
          completion_q.Push(std::move(item));
          continue;
        }

        // Copy output so post threads don't need access to RKNN buffers.
        size_t elems = runners[static_cast<size_t>(w)].Output0Elems();
        if (elems == 0) {
          elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        }
        item.out.resize(elems);
        std::memcpy(item.out.data(), out0, elems * sizeof(float));
        runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);

        const auto t_done = std::chrono::high_resolution_clock::now();
        item.infer_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_done - t0)
                .count();
        item.ok = 1;
        completion_q.Push(std::move(item));
      }
    });
  }

  // Submit threads: generate job ids, push into bounded in-flight queue.
  std::array<std::thread, kSubmitThreads> submit_threads;
  for (int s = 0; s < kSubmitThreads; ++s) {
    submit_threads[static_cast<size_t>(s)] = std::thread([&, s]() {
      PinThisThreadToCpu(submit_cpus[static_cast<size_t>(s)]);
      while (true) {
        const int job_id = next_job.fetch_add(1);
        if (job_id >= runs) {
          break;
        }
        if (!inflight_q.Push(job_id)) {
          break;
        }
      }
    });
  }

  const auto wall_t0 = std::chrono::high_resolution_clock::now();

  for (auto &t : submit_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  inflight_q.Close();

  for (auto &t : npu_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  completion_q.Stop();

  for (auto &t : post_threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  const auto wall_t1 = std::chrono::high_resolution_clock::now();

  r.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0)
                  .count();

  long long infer_us_sum = 0;
  long long post_us_sum = 0;
  std::vector<long long> infer_us_list;
  std::vector<long long> post_us_list;
  std::vector<long long> total_us_list;
  infer_us_list.reserve(static_cast<size_t>(runs));
  post_us_list.reserve(static_cast<size_t>(runs));
  total_us_list.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    const JobResult &jr = results[static_cast<size_t>(i)];
    infer_us_sum += jr.infer_us;
    post_us_sum += jr.post_us;
    infer_us_list.push_back(jr.infer_us);
    post_us_list.push_back(jr.post_us);
    total_us_list.push_back(jr.infer_us + jr.post_us);
  }
  r.avg_infer_ms = (infer_us_sum / 1000.0) / runs;
  r.avg_post_ms = (post_us_sum / 1000.0) / runs;
  r.avg_total_ms = r.avg_infer_ms + r.avg_post_ms;
  r.last_det_count = results.empty() ? 0 : results.back().det_count;
  r.infer_stats = ComputeLatStats(std::move(infer_us_list));
  r.post_stats = ComputeLatStats(std::move(post_us_list));
  r.total_stats = ComputeLatStats(std::move(total_us_list));

  if (print_ordered_per_run) {
    for (int i = 0; i < runs; ++i) {
      const JobResult &jr = results[static_cast<size_t>(i)];
      std::cout << "run=" << i << " ok=" << jr.ok << " det=" << jr.det_count
                << " infer_ms=" << (jr.infer_us / 1000.0)
                << " post_ms=" << (jr.post_us / 1000.0) << "\n";
    }
  }

  return r;
}

BenchResult RunPipeline6NpuWorkersPinned(const std::string &model_path,
                                        const cv::Mat &img_rgb, int rows,
                                        int cols, int detect_color, int runs,
                                        bool print_ordered_per_run) {
  BenchResult r;
  r.runs = runs;

  constexpr int kSubmitThreads = 3;
  constexpr int kNpuWorkers = 6;
  constexpr int kPostThreads = 1;
  constexpr size_t kInflightDepth = 8; // 6~8

  // Submit threads bound to CPU1/2/3, post threads bound to CPU7.
  const std::array<int, kSubmitThreads> submit_cpus = {1, 2, 3};
  const std::array<int, kPostThreads> post_cpus = {7};

  // 6 NPU workers pinned to CPU4/5/6 (cycle).
  const std::array<int, kNpuWorkers> npu_cpus = {4, 5, 6, 4, 5, 6};

  // 6 NPU contexts bound to NPU0/1/2 (cycle).
  const std::array<rknn_core_mask, kNpuWorkers> npu_masks = {
      RKNN_NPU_CORE_0,
      RKNN_NPU_CORE_1,
      RKNN_NPU_CORE_2,
      RKNN_NPU_CORE_AUTO,
      RKNN_NPU_CORE_AUTO,
      RKNN_NPU_CORE_AUTO,
  };

  std::array<RknnRunner, kNpuWorkers> runners;
  for (int i = 0; i < kNpuWorkers; ++i) {
    if (!runners[static_cast<size_t>(i)].Init(
            model_path, npu_masks[static_cast<size_t>(i)])) {
      throw std::runtime_error("Init failed for one of NPU cores");
    }
  }

  BoundedQueue<int> inflight_q(kInflightDepth);
  PostQueue completion_q;
  std::vector<JobResult> results(static_cast<size_t>(runs));

  std::atomic<int> next_job{0};

  // Post threads
  std::array<std::thread, kPostThreads> post_threads;
  for (int p = 0; p < kPostThreads; ++p) {
    post_threads[static_cast<size_t>(p)] = std::thread([&, p]() {
      PinThisThreadToCpu(post_cpus[static_cast<size_t>(p)]);
      std::vector<DetObj> dets;
      PostItem item;
      while (completion_q.Pop(item)) {
        JobResult jr;
        jr.infer_us = item.infer_us;
        jr.ok = item.ok;

        long long post_us = 0;
        size_t det_count = 0;
        if (item.ok && !item.out.empty()) {
          try {
            const auto t1 = std::chrono::high_resolution_clock::now();
            PostprocessSingleHead(item.out.data(), rows, cols, detect_color,
                                  kConfThresh, kNmsThresh, dets);
            const auto t2 = std::chrono::high_resolution_clock::now();
            post_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                    .count();
            det_count = dets.size();
          } catch (const std::exception &e) {
            std::cerr << "Postprocess error: " << e.what() << "\n";
            jr.ok = 0;
          }
        }

        jr.post_us = post_us;
        jr.det_count = det_count;
        if (item.job_id >= 0 && item.job_id < runs) {
          results[static_cast<size_t>(item.job_id)] = jr;
        }
      }
    });
  }

  // NPU worker threads
  std::array<std::thread, kNpuWorkers> npu_threads;
  for (int w = 0; w < kNpuWorkers; ++w) {
    npu_threads[static_cast<size_t>(w)] = std::thread([&, w]() {
      PinThisThreadToCpu(npu_cpus[static_cast<size_t>(w)]);
      std::vector<rknn_output> outputs;
      int job_id = -1;
      while (inflight_q.Pop(job_id)) {
        PostItem item;
        item.job_id = job_id;

        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!runners[static_cast<size_t>(w)].Infer(img_rgb, outputs)) {
          runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
          const auto t_fail = std::chrono::high_resolution_clock::now();
          item.infer_us =
              std::chrono::duration_cast<std::chrono::microseconds>(t_fail - t0)
                  .count();
          item.ok = 0;
          completion_q.Push(std::move(item));
          continue;
        }

        const float *out0 = runners[static_cast<size_t>(w)].Output0FloatPtr();
        if (!out0) {
          out0 = reinterpret_cast<const float *>(outputs.empty() ? nullptr : outputs[0].buf);
        }
        if (!out0) {
          runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
          const auto t_null = std::chrono::high_resolution_clock::now();
          item.infer_us =
              std::chrono::duration_cast<std::chrono::microseconds>(t_null - t0)
                  .count();
          item.ok = 0;
          completion_q.Push(std::move(item));
          continue;
        }

        size_t elems = runners[static_cast<size_t>(w)].Output0Elems();
        if (elems == 0) {
          elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        }
        item.out.resize(elems);
        std::memcpy(item.out.data(), out0, elems * sizeof(float));
        runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);

        const auto t_done = std::chrono::high_resolution_clock::now();
        item.infer_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_done - t0)
                .count();
        item.ok = 1;
        completion_q.Push(std::move(item));
      }
    });
  }

  // Submit threads
  std::array<std::thread, kSubmitThreads> submit_threads;
  for (int s = 0; s < kSubmitThreads; ++s) {
    submit_threads[static_cast<size_t>(s)] = std::thread([&, s]() {
      PinThisThreadToCpu(submit_cpus[static_cast<size_t>(s)]);
      while (true) {
        const int job_id = next_job.fetch_add(1);
        if (job_id >= runs) {
          break;
        }
        if (!inflight_q.Push(job_id)) {
          break;
        }
      }
    });
  }

  const auto wall_t0 = std::chrono::high_resolution_clock::now();

  for (auto &t : submit_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  inflight_q.Close();

  for (auto &t : npu_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  completion_q.Stop();

  for (auto &t : post_threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  const auto wall_t1 = std::chrono::high_resolution_clock::now();

  r.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0)
                  .count();

  long long infer_us_sum = 0;
  long long post_us_sum = 0;
  std::vector<long long> infer_us_list;
  std::vector<long long> post_us_list;
  std::vector<long long> total_us_list;
  infer_us_list.reserve(static_cast<size_t>(runs));
  post_us_list.reserve(static_cast<size_t>(runs));
  total_us_list.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    const JobResult &jr = results[static_cast<size_t>(i)];
    infer_us_sum += jr.infer_us;
    post_us_sum += jr.post_us;
    infer_us_list.push_back(jr.infer_us);
    post_us_list.push_back(jr.post_us);
    total_us_list.push_back(jr.infer_us + jr.post_us);
  }
  r.avg_infer_ms = (infer_us_sum / 1000.0) / runs;
  r.avg_post_ms = (post_us_sum / 1000.0) / runs;
  r.avg_total_ms = r.avg_infer_ms + r.avg_post_ms;
  r.last_det_count = results.empty() ? 0 : results.back().det_count;
  r.infer_stats = ComputeLatStats(std::move(infer_us_list));
  r.post_stats = ComputeLatStats(std::move(post_us_list));
  r.total_stats = ComputeLatStats(std::move(total_us_list));

  if (print_ordered_per_run) {
    for (int i = 0; i < runs; ++i) {
      const JobResult &jr = results[static_cast<size_t>(i)];
      std::cout << "run=" << i << " ok=" << jr.ok << " det=" << jr.det_count
                << " infer_ms=" << (jr.infer_us / 1000.0)
                << " post_ms=" << (jr.post_us / 1000.0) << "\n";
    }
  }

  return r;
}

BenchResult RunParallelMasked(const std::string &model_path,
                             const cv::Mat &img_rgb, int rows, int cols,
                             int detect_color, int runs,
                             bool print_ordered_per_run,
                             const std::vector<rknn_core_mask> &masks) {
  BenchResult r;
  r.runs = runs;

  const int workers = static_cast<int>(masks.size());
  if (workers <= 0) {
    throw std::runtime_error("No workers configured");
  }

  std::vector<RknnRunner> runners(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i) {
    if (!runners[static_cast<size_t>(i)].Init(model_path, masks[static_cast<size_t>(i)])) {
      throw std::runtime_error("Init failed for one of NPU cores");
    }
  }

  JobQueue jq;
  std::vector<JobResult> results(static_cast<size_t>(runs));

  std::vector<std::thread> threads(static_cast<size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    threads[static_cast<size_t>(w)] = std::thread([&, w]() {
      std::vector<rknn_output> outputs;
      std::vector<DetObj> dets;
      int job_id = -1;
      while (jq.Pop(job_id)) {
        if (job_id < 0 || job_id >= runs) {
          continue;
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!runners[static_cast<size_t>(w)].Infer(img_rgb, outputs)) {
          runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
          results[static_cast<size_t>(job_id)] = JobResult{};
          continue;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        const float *out0 = runners[static_cast<size_t>(w)].Output0FloatPtr();
        if (!out0) {
          out0 = reinterpret_cast<const float *>(outputs.empty() ? nullptr : outputs[0].buf);
        }
        if (out0) {
          PostprocessSingleHead(out0, rows, cols, detect_color, kConfThresh,
                                kNmsThresh, dets);
        }
        const auto t2 = std::chrono::high_resolution_clock::now();

        const long long infer_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        const long long post_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count();

        JobResult jr;
        jr.infer_us = infer_us;
        jr.post_us = post_us;
        jr.det_count = out0 ? dets.size() : 0;
        jr.ok = out0 ? 1 : 0;
        results[static_cast<size_t>(job_id)] = jr;

        runners[static_cast<size_t>(w)].ReleaseOutputs(outputs);
      }
    });
  }

  const auto wall_t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; ++i) {
    jq.Push(i);
  }
  jq.Stop();

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  const auto wall_t1 = std::chrono::high_resolution_clock::now();

  r.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0)
                  .count();

  long long infer_us_sum = 0;
  long long post_us_sum = 0;
  std::vector<long long> infer_us_list;
  std::vector<long long> post_us_list;
  std::vector<long long> total_us_list;
  infer_us_list.reserve(static_cast<size_t>(runs));
  post_us_list.reserve(static_cast<size_t>(runs));
  total_us_list.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    const JobResult &jr = results[static_cast<size_t>(i)];
    infer_us_sum += jr.infer_us;
    post_us_sum += jr.post_us;
    infer_us_list.push_back(jr.infer_us);
    post_us_list.push_back(jr.post_us);
    total_us_list.push_back(jr.infer_us + jr.post_us);
  }
  r.avg_infer_ms = (infer_us_sum / 1000.0) / runs;
  r.avg_post_ms = (post_us_sum / 1000.0) / runs;
  r.avg_total_ms = r.avg_infer_ms + r.avg_post_ms;
  r.last_det_count = results.empty() ? 0 : results.back().det_count;
  r.infer_stats = ComputeLatStats(std::move(infer_us_list));
  r.post_stats = ComputeLatStats(std::move(post_us_list));
  r.total_stats = ComputeLatStats(std::move(total_us_list));

  if (print_ordered_per_run) {
    for (int i = 0; i < runs; ++i) {
      const JobResult &jr = results[static_cast<size_t>(i)];
      std::cout << "run=" << i << " ok=" << jr.ok << " det=" << jr.det_count
                << " infer_ms=" << (jr.infer_us / 1000.0)
                << " post_ms=" << (jr.post_us / 1000.0) << "\n";
    }
  }

  return r;
}

BenchResult RunParallel3Core(const std::string &model_path,
                             const cv::Mat &img_rgb, int rows, int cols,
                             int detect_color, int runs,
                             bool print_ordered_per_run) {
  const std::vector<rknn_core_mask> masks = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1,
                                            RKNN_NPU_CORE_2};
  return RunParallelMasked(model_path, img_rgb, rows, cols, detect_color, runs,
                           print_ordered_per_run, masks);
}

BenchResult RunParallel6Workers012012(const std::string &model_path,
                                      const cv::Mat &img_rgb, int rows,
                                      int cols, int detect_color, int runs,
                                      bool print_ordered_per_run) {
  const std::vector<rknn_core_mask> masks = {
      RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
      RKNN_NPU_CORE_AUTO, RKNN_NPU_CORE_AUTO, RKNN_NPU_CORE_AUTO,
  };
  return RunParallelMasked(model_path, img_rgb, rows, cols, detect_color, runs,
                           print_ordered_per_run, masks);
}

BenchResult RunParallel6Workers012Auto2(const std::string &model_path,
                     const cv::Mat &img_rgb, int rows,
                     int cols, int detect_color, int runs,
                     bool print_ordered_per_run) {
  const std::vector<rknn_core_mask> masks = {
    RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0, RKNN_NPU_CORE_AUTO, RKNN_NPU_CORE_AUTO,
  };
  return RunParallelMasked(model_path, img_rgb, rows, cols, detect_color, runs,
               print_ordered_per_run, masks);
}

BenchResult RunParallel6Workers012Auto1(const std::string &model_path,
                     const cv::Mat &img_rgb, int rows,
                     int cols, int detect_color, int runs,
                     bool print_ordered_per_run) {
  const std::vector<rknn_core_mask> masks = {
    RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_AUTO,
  };
  return RunParallelMasked(model_path, img_rgb, rows, cols, detect_color, runs,
               print_ordered_per_run, masks);
}

void SaveOutputTxt(const std::string &path, const float *out, int rows,
                   int cols) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output txt: " << path << "\n";
    return;
  }
  for (int r = 0; r < rows; ++r) {
    const float *p = out + r * cols;
    for (int c = 0; c < cols; ++c) {
      ofs << p[c];
      if (c + 1 < cols) {
        ofs << ' ';
      }
    }
    ofs << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path = "../modles/False_none_0708.rknn";
  std::string image_path = "../1.png";
  int detect_color = -1; // -1 keep all, 0 blue-only, 1 red-only
  int runs = 1000;
  bool print_ordered_per_run = false;

  if (argc >= 2) {
    model_path = argv[1];
  }
  if (argc >= 3) {
    image_path = argv[2];
  }
  if (argc >= 4) {
    detect_color = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    runs = std::max(1, std::stoi(argv[4]));
  }
  if (argc >= 6) {
    print_ordered_per_run = (std::stoi(argv[5]) != 0);
  }

  RknnRunner runner;
  if (!runner.Init(model_path)) {
    return 2;
  }

  cv::Mat img_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img_bgr.empty()) {
    std::cerr << "Failed to read image: " << image_path << "\n";
    return 3;
  }

  // Preprocess once for fair benchmarking.
  cv::Mat vis_bgr;
  cv::resize(img_bgr, vis_bgr, cv::Size(kImgSize, kImgSize), 0, 0,
             cv::INTER_LINEAR);
  cv::Mat img_rgb;
  cv::cvtColor(vis_bgr, img_rgb, cv::COLOR_BGR2RGB);

  // Output shape (best-effort)
  const rknn_tensor_attr &out_attr0 = runner.OutputAttr(0);
  int rows = 0;
  int cols = 0;
  if (out_attr0.n_dims >= 3) {
    rows = static_cast<int>(out_attr0.dims[1]);
    cols = static_cast<int>(out_attr0.dims[2]);
  }
  if (rows <= 0 || cols <= 0) {
    rows = 25200;
    cols = 22;
  }

  try {
    // const BenchResult seq =
    //     RunSequential(runner, img_rgb, rows, cols, detect_color, runs);

    const BenchResult par = RunParallel3Core(model_path, img_rgb, rows, cols,
                         detect_color, runs,
                         print_ordered_per_run);

    // const BenchResult par6 = RunParallel6Workers012012(model_path, img_rgb,
    //                            rows, cols,
    //                            detect_color, runs,
    //                            print_ordered_per_run);

    // const BenchResult par6_auto2 = RunParallel6Workers012Auto2(model_path, img_rgb,
    //                            rows, cols,
    //                            detect_color, runs,
    //                            print_ordered_per_run);

    // const BenchResult par6_auto1 = RunParallel6Workers012Auto1(model_path, img_rgb,
    //                            rows, cols,
    //                            detect_color, runs,
    //                            print_ordered_per_run);

    // const BenchResult pipe = RunPipeline3Submit2PostPinned(model_path, img_rgb,
    //                              rows, cols,
    //                              detect_color, runs,
    //                              print_ordered_per_run);

    // const BenchResult pipe6 = RunPipeline6NpuWorkersPinned(model_path, img_rgb,
    //                 rows, cols,
    //                 detect_color, runs,
    //                 print_ordered_per_run);

    // std::cout << "Detections (last run): " << seq.last_det_count << "\n";

    // std::cout << "\n[Sequential, 1 context]\n";
    // std::cout << "wall: " << seq.wall_ms << " ms\n";
    // std::cout << "avg infer: " << seq.avg_infer_ms << " ms\n";
    // std::cout << "avg post:  " << seq.avg_post_ms << " ms\n";
    // std::cout << "avg total: " << seq.avg_total_ms << " ms\n";
    // std::cout << "infer avg/max/p90/p99: " << seq.infer_stats.avg_ms << " / "
    //       << seq.infer_stats.max_ms << " / " << seq.infer_stats.p90_ms
    //       << " / " << seq.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << seq.post_stats.avg_ms << " / "
    //       << seq.post_stats.max_ms << " / " << seq.post_stats.p90_ms
    //       << " / " << seq.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << seq.total_stats.avg_ms << " / "
    //       << seq.total_stats.max_ms << " / " << seq.total_stats.p90_ms
    //       << " / " << seq.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Parallel, 3 workers on NPU0/1/2]\n";
    // std::cout << "wall: " << par.wall_ms << " ms\n";
    // std::cout << "avg infer (per task CPU-side): " << par.avg_infer_ms
    //           << " ms\n";
    // std::cout << "avg post  (per task):         " << par.avg_post_ms
    //           << " ms\n";
    // std::cout << "avg total (per task):         " << par.avg_total_ms
    //           << " ms\n";
    // std::cout << "throughput: "
    //           << (runs / (par.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / par.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << par.infer_stats.avg_ms << " / "
    //       << par.infer_stats.max_ms << " / " << par.infer_stats.p90_ms
    //       << " / " << par.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << par.post_stats.avg_ms << " / "
    //       << par.post_stats.max_ms << " / " << par.post_stats.p90_ms
    //       << " / " << par.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << par.total_stats.avg_ms << " / "
    //       << par.total_stats.max_ms << " / " << par.total_stats.p90_ms
    //       << " / " << par.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Parallel, 6 workers on NPU0/1/2 + AUTOx3]\n";
    // std::cout << "wall: " << par6.wall_ms << " ms\n";
    // std::cout << "avg infer (per task CPU-side): " << par6.avg_infer_ms
    //           << " ms\n";
    // std::cout << "avg post  (per task):         " << par6.avg_post_ms
    //           << " ms\n";
    // std::cout << "avg total (per task):         " << par6.avg_total_ms
    //           << " ms\n";
    // std::cout << "throughput: "
    //           << (runs / (par6.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / par6.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << par6.infer_stats.avg_ms << " / "
    //       << par6.infer_stats.max_ms << " / " << par6.infer_stats.p90_ms
    //       << " / " << par6.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << par6.post_stats.avg_ms << " / "
    //       << par6.post_stats.max_ms << " / " << par6.post_stats.p90_ms
    //       << " / " << par6.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << par6.total_stats.avg_ms << " / "
    //       << par6.total_stats.max_ms << " / " << par6.total_stats.p90_ms
    //       << " / " << par6.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Parallel, 6 workers on NPU0/1/2 + AUTOx2]\n";
    // std::cout << "wall: " << par6_auto2.wall_ms << " ms\n";
    // std::cout << "avg infer (per task CPU-side): " << par6_auto2.avg_infer_ms
    //       << " ms\n";
    // std::cout << "avg post  (per task):         " << par6_auto2.avg_post_ms
    //       << " ms\n";
    // std::cout << "avg total (per task):         " << par6_auto2.avg_total_ms
    //       << " ms\n";
    // std::cout << "throughput: "
    //       << (runs / (par6_auto2.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / par6_auto2.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << par6_auto2.infer_stats.avg_ms << " / "
    //       << par6_auto2.infer_stats.max_ms << " / " << par6_auto2.infer_stats.p90_ms
    //       << " / " << par6_auto2.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << par6_auto2.post_stats.avg_ms << " / "
    //       << par6_auto2.post_stats.max_ms << " / " << par6_auto2.post_stats.p90_ms
    //       << " / " << par6_auto2.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << par6_auto2.total_stats.avg_ms << " / "
    //       << par6_auto2.total_stats.max_ms << " / " << par6_auto2.total_stats.p90_ms
    //       << " / " << par6_auto2.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Parallel, 6 workers on NPU0/1/2 + AUTOx1]\n";
    // std::cout << "wall: " << par6_auto1.wall_ms << " ms\n";
    // std::cout << "avg infer (per task CPU-side): " << par6_auto1.avg_infer_ms
    //       << " ms\n";
    // std::cout << "avg post  (per task):         " << par6_auto1.avg_post_ms
    //       << " ms\n";
    // std::cout << "avg total (per task):         " << par6_auto1.avg_total_ms
    //       << " ms\n";
    // std::cout << "throughput: "
    //       << (runs / (par6_auto1.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / par6_auto1.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << par6_auto1.infer_stats.avg_ms << " / "
    //       << par6_auto1.infer_stats.max_ms << " / " << par6_auto1.infer_stats.p90_ms
    //       << " / " << par6_auto1.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << par6_auto1.post_stats.avg_ms << " / "
    //       << par6_auto1.post_stats.max_ms << " / " << par6_auto1.post_stats.p90_ms
    //       << " / " << par6_auto1.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << par6_auto1.total_stats.avg_ms << " / "
    //       << par6_auto1.total_stats.max_ms << " / " << par6_auto1.total_stats.p90_ms
    //       << " / " << par6_auto1.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Pipeline pinned, submit CPU1/2/3, post CPU7, NPU workers=3 on CPU4/5/6, NPU0/1/2]\n";
    // std::cout << "wall: " << pipe.wall_ms << " ms\n";
    // std::cout << "avg submit(infer+copy): " << pipe.avg_infer_ms << " ms\n";
    // std::cout << "avg post:              " << pipe.avg_post_ms << " ms\n";
    // std::cout << "avg total:             " << pipe.avg_total_ms << " ms\n";
    // std::cout << "throughput: "
    //           << (runs / (pipe.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / pipe.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << pipe.infer_stats.avg_ms << " / "
    //       << pipe.infer_stats.max_ms << " / " << pipe.infer_stats.p90_ms
    //       << " / " << pipe.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << pipe.post_stats.avg_ms << " / "
    //       << pipe.post_stats.max_ms << " / " << pipe.post_stats.p90_ms
    //       << " / " << pipe.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << pipe.total_stats.avg_ms << " / "
    //       << pipe.total_stats.max_ms << " / " << pipe.total_stats.p90_ms
    //       << " / " << pipe.total_stats.p99_ms << " ms\n";

    // std::cout << "\n[Pipeline pinned, submit CPU1/2/3, post CPU7, NPU workers=6 on CPU4/5/6, NPU0/1/2 + AUTOx3]\n";
    // std::cout << "wall: " << pipe6.wall_ms << " ms\n";
    // std::cout << "avg submit(infer+copy): " << pipe6.avg_infer_ms << " ms\n";
    // std::cout << "avg post:              " << pipe6.avg_post_ms << " ms\n";
    // std::cout << "avg total:             " << pipe6.avg_total_ms << " ms\n";
    // std::cout << "throughput: "
    //           << (runs / (pipe6.wall_ms / 1000.0)) << " img/s\n";
    // std::cout << "speedup (wall): " << (seq.wall_ms / pipe6.wall_ms) << "x\n";
    // std::cout << "infer avg/max/p90/p99: " << pipe6.infer_stats.avg_ms << " / "
    //       << pipe6.infer_stats.max_ms << " / " << pipe6.infer_stats.p90_ms
    //       << " / " << pipe6.infer_stats.p99_ms << " ms\n";
    // std::cout << "post  avg/max/p90/p99: " << pipe6.post_stats.avg_ms << " / "
    //       << pipe6.post_stats.max_ms << " / " << pipe6.post_stats.p90_ms
    //       << " / " << pipe6.post_stats.p99_ms << " ms\n";
    // std::cout << "total avg/max/p90/p99: " << pipe6.total_stats.avg_ms << " / "
    //       << pipe6.total_stats.max_ms << " / " << pipe6.total_stats.p90_ms
    //       << " / " << pipe6.total_stats.p99_ms << " ms\n";
  } catch (const std::exception &e) {
    std::cerr << "Benchmark error: " << e.what() << "\n";
    return 6;
  }
  
  // Visualization (optional)
  // for (const auto &obj : dets) {
  //   cv::rectangle(vis_bgr, obj.rect, cv::Scalar(0, 255, 0), 2);
  //   for (const auto &pt : obj.kpts) {
  //     cv::circle(vis_bgr,
  //                cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 2,
  //                cv::Scalar(0, 0, 255), -1);
  //   }
  //   cv::putText(vis_bgr,
  //               "cid=" + std::to_string(obj.color_id) +
  //                   " cls=" + std::to_string(obj.label) +
  //                   " p=" + cv::format("%.2f", obj.prob),
  //               cv::Point(obj.rect.x, std::max(0, obj.rect.y - 6)),
  //               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  // }
  // cv::imwrite("result_rknn_cpp.jpg", vis_bgr);

  return 0;
}
