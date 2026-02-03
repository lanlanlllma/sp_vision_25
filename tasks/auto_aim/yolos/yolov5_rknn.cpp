#include "yolov5_rknn.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
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
}

YOLOV5_RKNN::~YOLOV5_RKNN()
{
  if (ctx_ != 0) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
}

std::list<Armor> YOLOV5_RKNN::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

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

  std::vector<rknn_output> outputs;
  if (!infer(input_rgb, outputs)) {
    release_outputs(outputs);
    return std::list<Armor>();
  }

  const float *out0 = nullptr;
  int rows = 0;
  int cols = 0;
  if (!output_attrs_.empty()) {
    const auto & out_attr = output_attrs_.front();
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
    release_outputs(outputs);
    return std::list<Armor>();
  }

  cv::Mat output(rows, cols, CV_32F, const_cast<float *>(out0));
  auto armors = parse(scale, output, raw_img, frame_count);

  release_outputs(outputs);
  return armors;
}

std::list<Armor> YOLOV5_RKNN::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classess
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    //颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     //color
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  //num
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
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
  cv::imshow("detection", detection);
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

  int ret = rknn_init(&ctx_, model_data.data(), model_data.size(), 0, nullptr);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_init failed, ret={}", ret);
    ctx_ = 0;
    return false;
  }

  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_query IN_OUT_NUM failed, ret={}", ret);
    return false;
  }

  input_attrs_.resize(io_num_.n_input);
  output_attrs_.resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_input; ++i) {
    input_attrs_[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
  }
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    output_attrs_[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
  }

  return true;
}

bool YOLOV5_RKNN::infer(const cv::Mat & img_rgb_u8, std::vector<rknn_output> & outputs)
{
  if (ctx_ == 0) {
    tools::logger()->error("RKNN context is null");
    return false;
  }
  if (img_rgb_u8.empty() || img_rgb_u8.type() != CV_8UC3) {
    tools::logger()->error("Expect RGB uint8 HWC CV_8UC3 input");
    return false;
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
    tools::logger()->error("rknn_inputs_set failed, ret={}", ret);
    return false;
  }

  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_run failed, ret={}", ret);
    return false;
  }

  outputs.assign(io_num_.n_output, {});
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    outputs[i].want_float = 1;
  }

  ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    tools::logger()->error("rknn_outputs_get failed, ret={}", ret);
    outputs.clear();
    return false;
  }

  return true;
}

void YOLOV5_RKNN::release_outputs(std::vector<rknn_output> & outputs)
{
  if (ctx_ == 0 || outputs.empty()) {
    return;
  }
  rknn_outputs_release(ctx_, outputs.size(), outputs.data());
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
