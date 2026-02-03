#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rknn_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
  std::vector<cv::Point2f> kpts; // 4 points
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

  std::vector<cv::Rect> boxes;
  std::vector<float> nms_scores;
  std::vector<DetObj> candidates;
  candidates.reserve(static_cast<size_t>(rows));

  for (int i = 0; i < rows; ++i) {
    const float *p = out + i * cols;

    float prob = Sigmoid(p[kConfIndex]);
    if (prob < conf_thresh) {
      continue;
    }

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

    obj.kpts.reserve(kNumKeypoints);
    for (int k = 0; k < kNumKeypoints; ++k) {
      float x = p[kKptStart + k * 2 + 0];
      float y = p[kKptStart + k * 2 + 1];
      obj.kpts.emplace_back(x, y);
    }

    float x_min = obj.kpts[0].x;
    float x_max = obj.kpts[0].x;
    float y_min = obj.kpts[0].y;
    float y_max = obj.kpts[0].y;
    for (int k = 1; k < kNumKeypoints; ++k) {
      x_min = std::min(x_min, obj.kpts[k].x);
      x_max = std::max(x_max, obj.kpts[k].x);
      y_min = std::min(y_min, obj.kpts[k].y);
      y_max = std::max(y_max, obj.kpts[k].y);
    }

    // Clip to input canvas (0..639)
    x_min = std::max(0.0f, std::min(x_min, static_cast<float>(kImgSize - 1)));
    y_min = std::max(0.0f, std::min(y_min, static_cast<float>(kImgSize - 1)));
    x_max = std::max(0.0f, std::min(x_max, static_cast<float>(kImgSize - 1)));
    y_max = std::max(0.0f, std::min(y_max, static_cast<float>(kImgSize - 1)));

    int left = static_cast<int>(std::floor(x_min));
    int top = static_cast<int>(std::floor(y_min));
    int right = static_cast<int>(std::ceil(x_max));
    int bottom = static_cast<int>(std::ceil(y_max));
    int w = std::max(0, right - left);
    int h = std::max(0, bottom - top);

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
    if (ctx_ != 0) {
      rknn_destroy(ctx_);
      ctx_ = 0;
    }
  }

  bool Init(const std::string &model_path) {
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

    return true;
  }

  void ReleaseOutputs(std::vector<rknn_output> &outputs) {
    if (ctx_ == 0 || outputs.empty()) {
      return;
    }
    rknn_outputs_release(ctx_, outputs.size(), outputs.data());
    outputs.clear();
  }

  const rknn_tensor_attr &OutputAttr(size_t idx) const {
    return output_attrs_.at(idx);
  }

private:
  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
};

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
  std::string model_path = "../0708.rknn";
  std::string image_path = "../1.png";
  int detect_color = -1; // -1 keep all, 0 blue-only, 1 red-only

  if (argc >= 2) {
    model_path = argv[1];
  }
  if (argc >= 3) {
    image_path = argv[2];
  }
  if (argc >= 4) {
    detect_color = std::stoi(argv[3]);
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

  cv::Mat vis_bgr;
  cv::resize(img_bgr, vis_bgr, cv::Size(kImgSize, kImgSize), 0, 0,
             cv::INTER_LINEAR);

  // Match onnx2rknn.py: feed RGB uint8 NHWC
  cv::Mat img_rgb;
  cv::cvtColor(vis_bgr, img_rgb, cv::COLOR_BGR2RGB);

  std::vector<rknn_output> outputs;
  if (!runner.Infer(img_rgb, outputs)) {
    runner.ReleaseOutputs(outputs);
    return 4;
  }

  // Expect output 0: [1, 25200, 22] float
  const rknn_tensor_attr &out_attr0 = runner.OutputAttr(0);

  // Best-effort parse dims (RKNN dims order depends on fmt; for typical
  // [1,25200,22] it's dims[0..2])
  int rows = 0;
  int cols = 0;
  if (out_attr0.n_dims >= 3) {
    rows = static_cast<int>(out_attr0.dims[1]);
    cols = static_cast<int>(out_attr0.dims[2]);
  }
  if (rows <= 0 || cols <= 0) {
    // fallback for unusual dim ordering
    rows = 25200;
    cols = 22;
  }

  const float *out0 = reinterpret_cast<const float *>(outputs[0].buf);
  if (!out0) {
    std::cerr << "Output buffer is null\n";
    runner.ReleaseOutputs(outputs);
    return 5;
  }

  // Save raw output like onnx2rknn.py / OpenvinoInfer.cpp
  SaveOutputTxt("rknn_yolov5_0_cpp.txt", out0, rows, cols);

  std::vector<DetObj> dets;
  try {
    PostprocessSingleHead(out0, rows, cols, detect_color, kConfThresh,
                          kNmsThresh, dets);
  } catch (const std::exception &e) {
    std::cerr << "Postprocess error: " << e.what() << "\n";
    runner.ReleaseOutputs(outputs);
    return 6;
  }

  std::cout << "Postprocess done, got " << dets.size() << " objects\n";

  // Visualization (optional)
  for (const auto &obj : dets) {
    cv::rectangle(vis_bgr, obj.rect, cv::Scalar(0, 255, 0), 2);
    for (const auto &pt : obj.kpts) {
      cv::circle(vis_bgr,
                 cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 2,
                 cv::Scalar(0, 0, 255), -1);
    }
    cv::putText(vis_bgr,
                "cid=" + std::to_string(obj.color_id) +
                    " cls=" + std::to_string(obj.label) +
                    " p=" + cv::format("%.2f", obj.prob),
                cv::Point(obj.rect.x, std::max(0, obj.rect.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  }
  cv::imwrite("result_rknn_cpp.jpg", vis_bgr);

  runner.ReleaseOutputs(outputs);
  return 0;
}
