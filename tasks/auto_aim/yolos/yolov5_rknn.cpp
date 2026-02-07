#include "yolov5_rknn.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
constexpr int kInputSize = 640;
constexpr int kOutputCols = 22;
constexpr int kOutputRowsFallback = 25200;

inline double SigmoidFast(double x)
{
  x = std::max(-50.0, std::min(50.0, x));
  return 1.0 / (1.0 + std::exp(-x));
}

inline double LogitClamp(double p)
{
  constexpr double kEps = 1e-6;
  p = std::max(kEps, std::min(1.0 - kEps, p));
  return std::log(p / (1.0 - p));
}
}  // namespace

YOLOV5_RKNN::YOLOV5_RKNN(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  if (yaml["yolov5_rknn_model_path"]) {
    model_path_ = yaml["yolov5_rknn_model_path"].as<std::string>();
  } else {
    model_path_ = yaml["yolov5_model_path"].as<std::string>();
  }

  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  if (!init_rknn(model_path_)) {
    throw std::runtime_error("Failed to init RKNN model: " + model_path_);
  }

  start_workers();
}

YOLOV5_RKNN::~YOLOV5_RKNN()
{
  stop_workers();
  for (auto & ctx_item : ctxs_) {
    if (ctx_item.ctx != 0) {
      rknn_destroy(ctx_item.ctx);
      ctx_item.ctx = 0;
    }
  }
}

std::list<Armor> YOLOV5_RKNN::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  const auto t_begin = std::chrono::steady_clock::now();

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(kInputSize) / bgr_img.rows;
  auto y_scale = static_cast<double>(kInputSize) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // preprocess (letterbox)
  auto input = cv::Mat(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

  cv::Mat input_rgb;
  cv::cvtColor(input, input_rgb, cv::COLOR_BGR2RGB);

  const auto t_pre_end = std::chrono::steady_clock::now();

  const auto t_submit = std::chrono::steady_clock::now();

  InferResult result;
  if (workers_started_) {
    InferJob job;
    job.input_rgb = input_rgb;
    auto fut = job.promise.get_future();
    {
      std::lock_guard<std::mutex> lk(queue_mu_);
      job_queue_.push_back(std::move(job));
    }
    queue_cv_.notify_one();
    result = fut.get();
  } else {
    const size_t ctx_index =
      static_cast<size_t>(next_ctx_.fetch_add(1, std::memory_order_relaxed) % kRknnContextCount);
    std::vector<rknn_output> outputs;
    if (!infer(input_rgb, outputs, ctx_index)) {
      release_outputs(outputs, ctx_index);
      return std::list<Armor>();
    }

    const float *out0 = nullptr;
    int rows = 0;
    int cols = 0;
    if (!ctxs_[ctx_index].output_attrs.empty()) {
      const auto & out_attr = ctxs_[ctx_index].output_attrs.front();
      if (out_attr.n_dims >= 3) {
        rows = static_cast<int>(out_attr.dims[1]);
        cols = static_cast<int>(out_attr.dims[2]);
      }
    }
    if (rows <= 0 || cols <= 0) {
      rows = kOutputRowsFallback;
      cols = kOutputCols;
    }

    if (!outputs.empty()) {
      out0 = reinterpret_cast<const float *>(outputs[0].buf);
    }
    if (!out0) {
      tools::logger()->error("RKNN output buffer is null");
      release_outputs(outputs, ctx_index);
      return std::list<Armor>();
    }

    result.ok = true;
    result.ctx_index = ctx_index;
    result.rows = rows;
    result.cols = cols;
    result.output.assign(out0, out0 + static_cast<size_t>(rows) * static_cast<size_t>(cols));
    release_outputs(outputs, ctx_index);
  }

  const auto t_infer_end = std::chrono::steady_clock::now();

  if (!result.ok || result.output.empty() || result.rows <= 0 || result.cols <= 0) {
    tools::logger()->error("RKNN output is empty");
    return std::list<Armor>();
  }

  cv::Mat output(result.rows, result.cols, CV_32F, result.output.data());
  auto armors = parse(scale, output, raw_img, frame_count);

  const auto t_post_end = std::chrono::steady_clock::now();

  if (debug_) {
    const auto pre_ms =
      std::chrono::duration<double, std::milli>(t_pre_end - t_begin).count();
    const auto wait_ms =
      std::chrono::duration<double, std::milli>(t_infer_end - t_submit).count();
    const auto infer_ms = result.infer_us / 1000.0;
    const auto post_ms =
      std::chrono::duration<double, std::milli>(t_post_end - t_infer_end).count();
    const auto total_ms =
      std::chrono::duration<double, std::milli>(t_post_end - t_begin).count();

    tools::logger()->debug(
      "[YOLOV5_RKNN] frame={} ctx={} pre={:.3f}ms wait={:.3f}ms infer={:.3f}ms post={:.3f}ms total={:.3f}ms rows={} cols={}",
      frame_count, result.ctx_index, pre_ms, wait_ms, infer_ms, post_ms, total_ms,
      result.rows, result.cols);
  }
  return armors;
}

std::list<Armor> YOLOV5_RKNN::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: kpts + conf + color(4) + num(9)
  thread_local std::vector<int> color_ids_buf;
  thread_local std::vector<int> num_ids_buf;
  thread_local std::vector<float> confidences_buf;
  thread_local std::vector<cv::Rect> boxes_buf;
  thread_local std::vector<std::vector<cv::Point2f>> armors_key_points_buf;

  auto & color_ids = color_ids_buf;
  auto & num_ids = num_ids_buf;
  auto & confidences = confidences_buf;
  auto & boxes = boxes_buf;
  auto & armors_key_points = armors_key_points_buf;

  color_ids.clear();
  num_ids.clear();
  confidences.clear();
  boxes.clear();
  armors_key_points.clear();

  const int rows = output.rows;
  const int cols = output.cols;
  if (rows <= 0 || cols <= 0) {
    return std::list<Armor>();
  }

  const float inv_scale = static_cast<float>(1.0 / scale);

  color_ids.reserve(rows);
  num_ids.reserve(rows);
  confidences.reserve(rows);
  boxes.reserve(rows);
  armors_key_points.reserve(rows);

  const double conf_logit_thresh = LogitClamp(score_threshold_);

  for (int r = 0; r < rows; ++r) {
    const float *p = output.ptr<float>(r);
    const double conf_logit = static_cast<double>(p[8]);
    if (conf_logit < conf_logit_thresh) {
      continue;
    }
    const float score = static_cast<float>(SigmoidFast(conf_logit));
    if (score < score_threshold_) {
      continue;
    }

    // color argmax (9..12)
    int color_id = 0;
    float best_color = p[9];
    for (int j = 10; j < 13; ++j) {
      if (p[j] > best_color) {
        best_color = p[j];
        color_id = j - 9;
      }
    }
    // num argmax (13..21)
    int class_id = 0;
    float best_num = p[13];
    for (int j = 14; j < 22; ++j) {
      if (p[j] > best_num) {
        best_num = p[j];
        class_id = j - 13;
      }
    }

    std::vector<cv::Point2f> armor_key_points;
    armor_key_points.resize(4);
    armor_key_points[0] = cv::Point2f(p[0] * inv_scale, p[1] * inv_scale);
    armor_key_points[1] = cv::Point2f(p[6] * inv_scale, p[7] * inv_scale);
    armor_key_points[2] = cv::Point2f(p[4] * inv_scale, p[5] * inv_scale);
    armor_key_points[3] = cv::Point2f(p[2] * inv_scale, p[3] * inv_scale);

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (size_t i = 1; i < armor_key_points.size(); ++i) {
      const auto & pt = armor_key_points[i];
      if (pt.x < min_x) min_x = pt.x;
      if (pt.x > max_x) max_x = pt.x;
      if (pt.y < min_y) min_y = pt.y;
      if (pt.y > max_y) max_y = pt.y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(color_id);
    num_ids.emplace_back(class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(std::move(armor_key_points));
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5_RKNN::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5_RKNN::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5_RKNN::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5_RKNN::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  //cv::imshow("detection", detection);
}

void YOLOV5_RKNN::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5_RKNN::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5_RKNN::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

bool YOLOV5_RKNN::init_rknn(const std::string & model_path)
{
  std::vector<uint8_t> model_data;
  if (!read_file(model_path, model_data)) {
    tools::logger()->error("Failed to read RKNN model: {}", model_path);
    return false;
  }

  const auto cleanup = [this]() {
    for (auto & ctx_item : ctxs_) {
      if (ctx_item.ctx != 0) {
        rknn_destroy(ctx_item.ctx);
        ctx_item.ctx = 0;
      }
      ctx_item.input_attrs.clear();
      ctx_item.output_attrs.clear();
      ctx_item.io_num = {};
    }
  };

  const std::array<rknn_core_mask, kRknnContextCount> masks = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
  };

  for (size_t i = 0; i < kRknnContextCount; ++i) {
    auto & ctx_item = ctxs_[i];
    int ret = rknn_init(&ctx_item.ctx, model_data.data(), model_data.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
      tools::logger()->error("rknn_init failed, ret={}", ret);
      ctx_item.ctx = 0;
      cleanup();
      return false;
    }

    ret = rknn_set_core_mask(ctx_item.ctx, masks[i]);
    if (ret != RKNN_SUCC) {
      tools::logger()->error("rknn_set_core_mask failed, ret={}", ret);
      cleanup();
      return false;
    }

    ret = rknn_query(ctx_item.ctx, RKNN_QUERY_IN_OUT_NUM, &ctx_item.io_num, sizeof(ctx_item.io_num));
    if (ret != RKNN_SUCC) {
      tools::logger()->error("rknn_query IN_OUT_NUM failed, ret={}", ret);
      cleanup();
      return false;
    }

    ctx_item.input_attrs.resize(ctx_item.io_num.n_input);
    ctx_item.output_attrs.resize(ctx_item.io_num.n_output);
    for (uint32_t j = 0; j < ctx_item.io_num.n_input; ++j) {
      ctx_item.input_attrs[j].index = j;
      rknn_query(
        ctx_item.ctx, RKNN_QUERY_INPUT_ATTR, &ctx_item.input_attrs[j],
        sizeof(rknn_tensor_attr));
    }
    for (uint32_t j = 0; j < ctx_item.io_num.n_output; ++j) {
      ctx_item.output_attrs[j].index = j;
      rknn_query(
        ctx_item.ctx, RKNN_QUERY_OUTPUT_ATTR, &ctx_item.output_attrs[j],
        sizeof(rknn_tensor_attr));
    }
  }

  return true;
}

void YOLOV5_RKNN::start_workers()
{
  if (workers_started_) {
    return;
  }
  stop_workers_ = false;
  for (size_t i = 0; i < kRknnContextCount; ++i) {
    workers_[i] = std::thread([this, i] { worker_loop(i); });
  }
  workers_started_ = true;
}

void YOLOV5_RKNN::stop_workers()
{
  if (!workers_started_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(queue_mu_);
    stop_workers_ = true;
  }
  queue_cv_.notify_all();
  for (auto & t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
  workers_started_ = false;
}

void YOLOV5_RKNN::worker_loop(size_t ctx_index)
{
  while (true) {
    InferJob job;
    {
      std::unique_lock<std::mutex> lk(queue_mu_);
      queue_cv_.wait(lk, [this] { return stop_workers_ || !job_queue_.empty(); });
      if (stop_workers_ && job_queue_.empty()) {
        return;
      }
      job = std::move(job_queue_.front());
      job_queue_.pop_front();
    }

    InferResult result;
    result.ctx_index = ctx_index;

    std::vector<rknn_output> outputs;
    const auto t0 = std::chrono::steady_clock::now();
    if (!infer(job.input_rgb, outputs, ctx_index)) {
      release_outputs(outputs, ctx_index);
      result.ok = false;
      result.infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0)
        .count();
      job.promise.set_value(std::move(result));
      continue;
    }

    const auto t1 = std::chrono::steady_clock::now();

    const float *out0 = nullptr;
    int rows = 0;
    int cols = 0;
    if (!ctxs_[ctx_index].output_attrs.empty()) {
      const auto & out_attr = ctxs_[ctx_index].output_attrs.front();
      if (out_attr.n_dims >= 3) {
        rows = static_cast<int>(out_attr.dims[1]);
        cols = static_cast<int>(out_attr.dims[2]);
      }
    }
    if (rows <= 0 || cols <= 0) {
      rows = kOutputRowsFallback;
      cols = kOutputCols;
    }

    if (!outputs.empty()) {
      out0 = reinterpret_cast<const float *>(outputs[0].buf);
    }

    if (out0) {
      result.ok = true;
      result.rows = rows;
      result.cols = cols;
      result.output.assign(out0, out0 + static_cast<size_t>(rows) * static_cast<size_t>(cols));
    } else {
      result.ok = false;
    }

    release_outputs(outputs, ctx_index);

    result.infer_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    job.promise.set_value(std::move(result));
  }
}

bool YOLOV5_RKNN::infer(
  const cv::Mat & img_rgb_u8, std::vector<rknn_output> & outputs, size_t ctx_index)
{
  if (ctx_index >= kRknnContextCount || ctxs_[ctx_index].ctx == 0) {
    tools::logger()->error("RKNN context is null");
    return false;
  }
  if (img_rgb_u8.empty() || img_rgb_u8.type() != CV_8UC3) {
    tools::logger()->error("Expect RGB uint8 HWC CV_8UC3 input");
    return false;
  }

  auto & ctx_item = ctxs_[ctx_index];

  rknn_input in;
  std::memset(&in, 0, sizeof(in));
  in.index = 0;
  in.type = RKNN_TENSOR_UINT8;
  in.fmt = RKNN_TENSOR_NHWC;
  in.size = static_cast<uint32_t>(img_rgb_u8.total() * img_rgb_u8.elemSize());
  in.buf = const_cast<unsigned char *>(img_rgb_u8.ptr<unsigned char>());

  int ret = rknn_inputs_set(ctx_item.ctx, ctx_item.io_num.n_input, &in);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_inputs_set failed, ret={}", ret);
    return false;
  }

  ret = rknn_run(ctx_item.ctx, nullptr);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_run failed, ret={}", ret);
    return false;
  }

  outputs.assign(ctx_item.io_num.n_output, {});
  for (uint32_t i = 0; i < ctx_item.io_num.n_output; ++i) {
    outputs[i].want_float = 1;
  }

  ret = rknn_outputs_get(ctx_item.ctx, ctx_item.io_num.n_output, outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_outputs_get failed, ret={}", ret);
    outputs.clear();
    return false;
  }

  return true;
}

void YOLOV5_RKNN::release_outputs(std::vector<rknn_output> & outputs, size_t ctx_index)
{
  if (ctx_index >= kRknnContextCount || ctxs_[ctx_index].ctx == 0 || outputs.empty()) {
    return;
  }
  rknn_outputs_release(ctxs_[ctx_index].ctx, outputs.size(), outputs.data());
  outputs.clear();
}

bool YOLOV5_RKNN::read_file(const std::string & path, std::vector<uint8_t> & data)
{
  FILE * fp = fopen(path.c_str(), "rb");
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

}  // namespace auto_aim
