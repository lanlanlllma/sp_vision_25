#include "yolov5_rknn.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim {
namespace {
    constexpr int kInputSize          = 640;
    constexpr int kOutputCols         = 22;
    constexpr int kOutputRowsFallback = 25200;

    // 3-branch decode constants (matching inference_example.py)
    constexpr int kNumAnchors  = 3;
    constexpr int kNumChannels = 22; // per anchor
    constexpr int kNumScales   = 3;

    // Anchor sizes: [[10,13],[16,30],[33,23], [30,61],[62,45],[59,119], [116,90],[156,198],[373,326]]
    static const float kAnchors[9][2] = {
        { 10.f, 13.f },  { 16.f, 30.f },  { 33.f, 23.f },   { 30.f, 61.f },   { 62.f, 45.f },
        { 59.f, 119.f }, { 116.f, 90.f }, { 156.f, 198.f }, { 373.f, 326.f },
    };
    static const int kMasks[3][3] = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };
    static const int kStrides[3]  = { 8, 16, 32 };

    // Per-context fallback output storage (only used when !use_io_mem).
    // Safe because each ctx_index is exclusively owned by one worker thread.
    std::array<std::vector<rknn_output>, 3> g_fallback_outputs;

    // Decode 3-branch RKNN outputs [1,66,H,W] into flat [25200, 22] vector.
    // Matches inference_example.py lines 519-577 exactly.
    // Cache-optimized: pre-computes strides to reduce random NCHW access overhead.
    static std::vector<float> decode_3branch(
        const std::array<const float*, 3>& output_ptrs,
        const std::vector<rknn_tensor_attr>& output_attrs
    ) {
        // Use thread_local to avoid repeated heap allocation across frames
        thread_local std::vector<float> merged;
        merged.clear();
        merged.reserve(kOutputRowsFallback * kOutputCols);

        for (int s = 0; s < kNumScales; ++s) {
            const float* data = output_ptrs[static_cast<size_t>(s)];
            // Determine H, W from output attrs
            int h = 0, w = 0;
            if (s < static_cast<int>(output_attrs.size()) && output_attrs[s].n_dims >= 4) {
                h = static_cast<int>(output_attrs[s].dims[2]);
                w = static_cast<int>(output_attrs[s].dims[3]);
            } else {
                // Fallback: 640/stride
                h = kInputSize / kStrides[s];
                w = kInputSize / kStrides[s];
            }

            const float stride = static_cast<float>(kStrides[s]);
            const size_t hw    = static_cast<size_t>(h) * static_cast<size_t>(w);

            // data layout: [1, 66, H, W] = [1, 3*22, H, W]  (NCHW)
            // We need to reshape to [3, 22, H, W] then transpose to [3, H, W, 22]
            // then decode keypoints and flatten to [3*H*W, 22]
            for (int a = 0; a < kNumAnchors; ++a) {
                const float anchor_w     = kAnchors[kMasks[s][a]][0];
                const float anchor_h     = kAnchors[kMasks[s][a]][1];
                const size_t anchor_base = static_cast<size_t>(a * kNumChannels) * hw;

                for (int gy = 0; gy < h; ++gy) {
                    for (int gx = 0; gx < w; ++gx) {
                        // Read 22 channels for this anchor at (gy, gx)
                        // In NCHW: channel c is at offset anchor_base + c*hw + spatial_idx
                        const size_t spatial_idx =
                            static_cast<size_t>(gy) * static_cast<size_t>(w) + gx;
                        float row[22];
                        for (int c = 0; c < kNumChannels; ++c) {
                            row[c] = data[anchor_base + static_cast<size_t>(c) * hw + spatial_idx];
                        }

                        // Decode keypoints (channels 0-7): kpt = raw * anchor_wh + grid * stride
                        // LINEAR decode, NOT sigmoid (matches Python reference)
                        for (int kp = 0; kp < 4; ++kp) {
                            row[kp * 2] = row[kp * 2] * anchor_w + static_cast<float>(gx) * stride;
                            row[kp * 2 + 1] =
                                row[kp * 2 + 1] * anchor_h + static_cast<float>(gy) * stride;
                        }

                        // Channels 8-21: pass-through (conf/color/num logits)
                        merged.insert(merged.end(), row, row + kNumChannels);
                    }
                }
            }
        }

        return merged;
    }

    inline double SigmoidFast(double x) {
        x = std::max(-50.0, std::min(50.0, x));
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double LogitClamp(double p) {
        constexpr double kEps = 1e-6;
        p                     = std::max(kEps, std::min(1.0 - kEps, p));
        return std::log(p / (1.0 - p));
    }
} // namespace

YOLOV5_RKNN::YOLOV5_RKNN(const std::string& config_path, bool debug):
    debug_(debug),
    detector_(config_path, false) {
    auto yaml = YAML::LoadFile(config_path);

    if (yaml["yolov5_rknn_model_path"]) {
        model_path_ = yaml["yolov5_rknn_model_path"].as<std::string>();
    } else {
        model_path_ = yaml["yolov5_model_path"].as<std::string>();
    }

    binary_threshold_ = yaml["threshold"].as<double>();
    min_confidence_   = yaml["min_confidence"].as<double>();
    int x = 0, y = 0, width = 0, height = 0;
    x                = yaml["roi"]["x"].as<int>();
    y                = yaml["roi"]["y"].as<int>();
    width            = yaml["roi"]["width"].as<int>();
    height           = yaml["roi"]["height"].as<int>();
    use_roi_         = yaml["use_roi"].as<bool>();
    use_traditional_ = yaml["use_traditional"].as<bool>();
    roi_             = cv::Rect(x, y, width, height);
    offset_          = cv::Point2f(x, y);

    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);

    if (!init_rknn(model_path_)) {
        throw std::runtime_error("Failed to init RKNN model: " + model_path_);
    }

    start_workers();
}

YOLOV5_RKNN::~YOLOV5_RKNN() {
    stop_workers();
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_jobs_.clear();
    }
    for (auto& ctx_item: ctxs_) {
        destroy_io_mem(ctx_item);
        if (ctx_item.ctx != 0) {
            rknn_destroy(ctx_item.ctx);
            ctx_item.ctx = 0;
        }
    }
}

std::list<Armor> YOLOV5_RKNN::detect(const cv::Mat& raw_img, int frame_count) {
    if (raw_img.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return std::list<Armor>();
    }

    const auto t_begin = std::chrono::steady_clock::now();

    cv::Mat input_rgb;
    double scale = 1.0;
    if (!preprocess(raw_img, input_rgb, scale)) {
        return std::list<Armor>();
    }

    const auto t_pre_end = std::chrono::steady_clock::now();

    const auto t_submit = std::chrono::steady_clock::now();

    auto fut           = enqueue_infer(input_rgb);
    InferResult result = fut.get();

    const auto t_infer_end = std::chrono::steady_clock::now();

    if (!result.ok || result.output.empty() || result.rows <= 0 || result.cols <= 0) {
        tools::logger()->error("RKNN output is empty");
        return std::list<Armor>();
    }

    cv::Mat output(result.rows, result.cols, CV_32F, result.output.data());
    std::list<Armor> armors;
    {
        std::lock_guard<std::mutex> lk(postprocess_mu_);
        armors = parse(scale, output, raw_img, frame_count);
    }

    const auto t_post_end = std::chrono::steady_clock::now();

    if (debug_) {
        const auto pre_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_begin).count();
        const auto wait_ms =
            std::chrono::duration<double, std::milli>(t_infer_end - t_submit).count();
        const auto infer_ms = result.infer_us / 1000.0;
        const auto post_ms =
            std::chrono::duration<double, std::milli>(t_post_end - t_infer_end).count();
        const auto total_ms =
            std::chrono::duration<double, std::milli>(t_post_end - t_begin).count();
        log_timing(
            frame_count,
            result.ctx_index,
            pre_ms,
            wait_ms,
            infer_ms,
            post_ms,
            total_ms,
            result.rows,
            result.cols
        );
    }
    return armors;
}

YOLOV5_RKNN::JobId YOLOV5_RKNN::submit(const cv::Mat& raw_img, int frame_count) {
    if (raw_img.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return kInvalidJobId;
    }

    const auto t_begin = std::chrono::steady_clock::now();

    cv::Mat input_rgb;
    double scale = 1.0;
    if (!preprocess(raw_img, input_rgb, scale)) {
        return kInvalidJobId;
    }

    const auto t_pre_end = std::chrono::steady_clock::now();

    auto fut = enqueue_infer(input_rgb);

    const auto t_submit = std::chrono::steady_clock::now();

    PendingJob pending;
    pending.scale       = scale;
    pending.raw_img     = raw_img.clone();
    pending.frame_count = frame_count;
    pending.future      = std::move(fut);
    pending.t_begin     = t_begin;
    pending.t_pre_end   = t_pre_end;
    pending.t_submit    = t_submit;

    const JobId job_id = next_job_id_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_jobs_.emplace(job_id, std::move(pending));
    }

    return job_id;
}

std::list<Armor> YOLOV5_RKNN::wait(JobId job_id) {
    PendingJob pending;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto it = pending_jobs_.find(job_id);
        if (it == pending_jobs_.end()) {
            tools::logger()->warn("Invalid job id: {}", job_id);
            return std::list<Armor>();
        }
        pending = std::move(it->second);
        pending_jobs_.erase(it);
    }

    InferResult result     = pending.future.get();
    const auto t_infer_end = std::chrono::steady_clock::now();
    if (!result.ok || result.output.empty() || result.rows <= 0 || result.cols <= 0) {
        tools::logger()->error("RKNN output is empty");
        return std::list<Armor>();
    }

    cv::Mat output(result.rows, result.cols, CV_32F, result.output.data());
    std::list<Armor> armors;
    {
        std::lock_guard<std::mutex> lk(postprocess_mu_);
        armors = parse(pending.scale, output, pending.raw_img, pending.frame_count);
    }
    const auto t_post_end = std::chrono::steady_clock::now();

    if (debug_) {
        const auto pre_ms =
            std::chrono::duration<double, std::milli>(pending.t_pre_end - pending.t_begin).count();
        const auto wait_ms =
            std::chrono::duration<double, std::milli>(t_infer_end - pending.t_submit).count();
        const auto infer_ms = result.infer_us / 1000.0;
        const auto post_ms =
            std::chrono::duration<double, std::milli>(t_post_end - t_infer_end).count();
        const auto total_ms =
            std::chrono::duration<double, std::milli>(t_post_end - pending.t_begin).count();
        log_timing(
            pending.frame_count,
            result.ctx_index,
            pre_ms,
            wait_ms,
            infer_ms,
            post_ms,
            total_ms,
            result.rows,
            result.cols
        );
    }
    return armors;
}

bool YOLOV5_RKNN::try_wait(JobId job_id, std::list<Armor>& armors) {
    PendingJob pending;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto it = pending_jobs_.find(job_id);
        if (it == pending_jobs_.end()) {
            return false;
        }
        if (it->second.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return false;
        }
        pending = std::move(it->second);
        pending_jobs_.erase(it);
    }

    InferResult result     = pending.future.get();
    const auto t_infer_end = std::chrono::steady_clock::now();
    if (!result.ok || result.output.empty() || result.rows <= 0 || result.cols <= 0) {
        tools::logger()->error("RKNN output is empty");
        armors.clear();
        return true;
    }

    cv::Mat output(result.rows, result.cols, CV_32F, result.output.data());
    {
        std::lock_guard<std::mutex> lk(postprocess_mu_);
        armors = parse(pending.scale, output, pending.raw_img, pending.frame_count);
    }

    const auto t_post_end = std::chrono::steady_clock::now();
    if (debug_) {
        const auto pre_ms =
            std::chrono::duration<double, std::milli>(pending.t_pre_end - pending.t_begin).count();
        const auto wait_ms =
            std::chrono::duration<double, std::milli>(t_infer_end - pending.t_submit).count();
        const auto infer_ms = result.infer_us / 1000.0;
        const auto post_ms =
            std::chrono::duration<double, std::milli>(t_post_end - t_infer_end).count();
        const auto total_ms =
            std::chrono::duration<double, std::milli>(t_post_end - pending.t_begin).count();
        log_timing(
            pending.frame_count,
            result.ctx_index,
            pre_ms,
            wait_ms,
            infer_ms,
            post_ms,
            total_ms,
            result.rows,
            result.cols
        );
    }
    return true;
}

std::list<Armor>
YOLOV5_RKNN::parse(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count) {
    // Postprocess matching inference_example.py::postprocess_single_head
    // Layout per row: [0-7] keypoints, [8] conf logit, [9-12] color scores, [13-21] num scores
    thread_local std::vector<int> color_ids_buf;
    thread_local std::vector<int> num_ids_buf;
    thread_local std::vector<float> confidences_buf;
    thread_local std::vector<float> score_num_buf;
    thread_local std::vector<cv::Rect> boxes_buf;
    thread_local std::vector<std::vector<cv::Point2f>> armors_key_points_buf;

    auto& color_ids         = color_ids_buf;
    auto& num_ids           = num_ids_buf;
    auto& confidences       = confidences_buf;
    auto& score_nums        = score_num_buf;
    auto& boxes             = boxes_buf;
    auto& armors_key_points = armors_key_points_buf;

    color_ids.clear();
    num_ids.clear();
    confidences.clear();
    score_nums.clear();
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
    score_nums.reserve(rows);
    boxes.reserve(rows);
    armors_key_points.reserve(rows);

    const double conf_logit_thresh = LogitClamp(score_threshold_);

    for (int r = 0; r < rows; ++r) {
        const float* p = output.ptr<float>(r);

        // Step 1: sigmoid(conf) + threshold
        const double conf_logit = static_cast<double>(p[8]);
        if (conf_logit < conf_logit_thresh) {
            continue;
        }
        const float score = static_cast<float>(SigmoidFast(conf_logit));
        if (score < score_threshold_) {
            continue;
        }

        // Step 2: color argmax (9..12)
        int color_id     = 0;
        float best_color = p[9];
        for (int j = 10; j < 13; ++j) {
            if (p[j] > best_color) {
                best_color = p[j];
                color_id   = j - 9;
            }
        }

        // Step 3: color filter — drop None(2)/Purple(3) per postprocess_single_head
        if (color_id == 2 || color_id == 3) {
            continue;
        }

        // Step 4: num argmax (13..21)
        int class_id   = 0;
        float best_num = p[13];
        for (int j = 14; j < 22; ++j) {
            if (p[j] > best_num) {
                best_num = p[j];
                class_id = j - 13;
            }
        }

        // Step 5: score_num threshold (matches Python: keep3 = score_num > conf_thresh)
        if (best_num <= score_threshold_) {
            continue;
        }

        // Step 6: keypoints + bbox
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
            const auto& pt = armor_key_points[i];
            if (pt.x < min_x)
                min_x = pt.x;
            if (pt.x > max_x)
                max_x = pt.x;
            if (pt.y < min_y)
                min_y = pt.y;
            if (pt.y > max_y)
                max_y = pt.y;
        }

        cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

        color_ids.emplace_back(color_id);
        num_ids.emplace_back(class_id);
        boxes.emplace_back(rect);
        confidences.emplace_back(score);
        score_nums.emplace_back(best_num);
        armors_key_points.emplace_back(std::move(armor_key_points));
    }

    // Step 7: NMS using score_num as NMS score (matches Python: nms_xyxy(boxes, score_num, ...))
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, score_nums, score_threshold_, nms_threshold_, indices);

    std::list<Armor> armors;
    for (const auto& i: indices) {
        if (use_roi_) {
            armors.emplace_back(
                color_ids[i],
                num_ids[i],
                confidences[i],
                boxes[i],
                armors_key_points[i],
                offset_
            );
        } else {
            armors.emplace_back(
                color_ids[i],
                num_ids[i],
                confidences[i],
                boxes[i],
                armors_key_points[i]
            );
        }
    }

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
        if (use_traditional_)
            detector_.detect(*it, bgr_img);

        it->center_norm = get_center_norm(bgr_img, it->center);
        ++it;
    }

    if (debug_)
        draw_detections(bgr_img, armors, frame_count);

    return armors;
}

bool YOLOV5_RKNN::check_name(const Armor& armor) const {
    auto name_ok       = armor.name != ArmorName::not_armor;
    auto confidence_ok = armor.confidence > min_confidence_;

    // 保存不确定的图案，用于神经网络的迭代
    // if (name_ok && !confidence_ok) save(armor);

    return name_ok && confidence_ok;
}

bool YOLOV5_RKNN::check_type(const Armor& armor) const {
    auto name_ok = (armor.type == ArmorType::small)
        ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
        : (armor.name != ArmorName::two && armor.name != ArmorName::sentry
           && armor.name != ArmorName::outpost);

    // 保存异常的图案，用于神经网络的迭代
    // if (!name_ok) save(armor);

    return name_ok;
}

cv::Point2f YOLOV5_RKNN::get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const {
    auto h = bgr_img.rows;
    auto w = bgr_img.cols;
    return { center.x / w, center.y / h };
}

void YOLOV5_RKNN::draw_detections(
    const cv::Mat& img,
    const std::list<Armor>& armors,
    int frame_count
) const {
    auto detection = img.clone();
    tools::draw_text(detection, fmt::format("[{}]", frame_count), { 10, 30 }, { 255, 255, 255 });
    for (const auto& armor: armors) {
        auto info = fmt::format(
            "{:.2f} {} {} {}",
            armor.confidence,
            COLORS[armor.color],
            ARMOR_NAMES[armor.name],
            ARMOR_TYPES[armor.type]
        );
        tools::draw_points(detection, armor.points, { 0, 255, 0 });
        tools::draw_text(detection, info, armor.center, { 0, 255, 0 });
    }

    if (use_roi_) {
        cv::Scalar green(0, 255, 0);
        cv::rectangle(detection, roi_, green, 2);
    }
    cv::resize(detection, detection, {}, 0.5, 0.5); // 显示时缩小图片尺寸
                                                    //cv::imshow("detection", detection);
}

void YOLOV5_RKNN::save(const Armor& armor, const cv::Mat& img) const {
    auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
    auto img_path  = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
    cv::imwrite(img_path, img);
}

std::list<Armor>
YOLOV5_RKNN::postprocess(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count) {
    return parse(scale, output, bgr_img, frame_count);
}

void YOLOV5_RKNN::log_timing(
    int frame_count,
    size_t ctx_index,
    double pre_ms,
    double wait_ms,
    double infer_ms,
    double post_ms,
    double total_ms,
    int rows,
    int cols
) const {
    tools::logger()->debug(
        "[YOLOV5_RKNN] frame={} ctx={} pre={:.3f}ms wait={:.3f}ms infer={:.3f}ms "
        "post={:.3f}ms total={:.3f}ms rows={} cols={}",
        frame_count,
        ctx_index,
        pre_ms,
        wait_ms,
        infer_ms,
        post_ms,
        total_ms,
        rows,
        cols
    );
}

bool YOLOV5_RKNN::init_rknn(const std::string& model_path) {
    std::vector<uint8_t> model_data;
    if (!read_file(model_path, model_data)) {
        tools::logger()->error("Failed to read RKNN model: {}", model_path);
        return false;
    }

    const auto cleanup = [this]() {
        for (auto& ctx_item: ctxs_) {
            destroy_io_mem(ctx_item);
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
        auto& ctx_item = ctxs_[i];
        int ret        = rknn_init(&ctx_item.ctx, model_data.data(), model_data.size(), 0, nullptr);
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

        ret = rknn_query(
            ctx_item.ctx,
            RKNN_QUERY_IN_OUT_NUM,
            &ctx_item.io_num,
            sizeof(ctx_item.io_num)
        );
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
                ctx_item.ctx,
                RKNN_QUERY_INPUT_ATTR,
                &ctx_item.input_attrs[j],
                sizeof(rknn_tensor_attr)
            );
        }
        for (uint32_t j = 0; j < ctx_item.io_num.n_output; ++j) {
            ctx_item.output_attrs[j].index = j;
            rknn_query(
                ctx_item.ctx,
                RKNN_QUERY_OUTPUT_ATTR,
                &ctx_item.output_attrs[j],
                sizeof(rknn_tensor_attr)
            );
        }

        // Initialize zero-copy IO memory for this context
        if (!init_io_mem(ctx_item)) {
            tools::logger()->warn("init_io_mem failed for ctx {}, falling back to copy mode", i);
            // Non-fatal: will use legacy rknn_inputs_set/rknn_outputs_get path
        }
    }

    return true;
}

bool YOLOV5_RKNN::init_io_mem(RknnContext& ctx_item) {
    destroy_io_mem(ctx_item);

    if (ctx_item.ctx == 0 || ctx_item.io_num.n_input < 1 || ctx_item.io_num.n_output < 1) {
        return false;
    }

    // Set up input memory
    ctx_item.io_input_attr      = ctx_item.input_attrs[0];
    ctx_item.io_input_attr.type = RKNN_TENSOR_UINT8;
    ctx_item.io_input_attr.fmt  = RKNN_TENSOR_NHWC;

    ctx_item.input_mem = rknn_create_mem(ctx_item.ctx, ctx_item.io_input_attr.size_with_stride);
    if (!ctx_item.input_mem) {
        tools::logger()->error("rknn_create_mem(input) failed");
        return false;
    }

    int ret = rknn_set_io_mem(ctx_item.ctx, ctx_item.input_mem, &ctx_item.io_input_attr);
    if (ret != RKNN_SUCC) {
        tools::logger()->error("rknn_set_io_mem(input) failed, ret={}", ret);
        rknn_destroy_mem(ctx_item.ctx, ctx_item.input_mem);
        ctx_item.input_mem = nullptr;
        return false;
    }

    // Set up output memories — request FLOAT32 for direct access
    ctx_item.output_mems.resize(ctx_item.io_num.n_output, nullptr);
    ctx_item.io_output_attrs = ctx_item.output_attrs;
    for (uint32_t i = 0; i < ctx_item.io_num.n_output; ++i) {
        ctx_item.io_output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        const int out_bytes =
            static_cast<int>(ctx_item.io_output_attrs[i].n_elems) * static_cast<int>(sizeof(float));
        ctx_item.output_mems[i] = rknn_create_mem(ctx_item.ctx, out_bytes);
        if (!ctx_item.output_mems[i]) {
            tools::logger()->error("rknn_create_mem(output[{}]) failed", i);
            destroy_io_mem(ctx_item);
            return false;
        }
        ret = rknn_set_io_mem(ctx_item.ctx, ctx_item.output_mems[i], &ctx_item.io_output_attrs[i]);
        if (ret != RKNN_SUCC) {
            tools::logger()->error("rknn_set_io_mem(output[{}]) failed, ret={}", i, ret);
            destroy_io_mem(ctx_item);
            return false;
        }
    }

    ctx_item.use_io_mem = true;
    return true;
}

void YOLOV5_RKNN::destroy_io_mem(RknnContext& ctx_item) {
    if (ctx_item.ctx == 0) {
        return;
    }
    if (ctx_item.input_mem) {
        rknn_destroy_mem(ctx_item.ctx, ctx_item.input_mem);
        ctx_item.input_mem = nullptr;
    }
    for (auto& m: ctx_item.output_mems) {
        if (m) {
            rknn_destroy_mem(ctx_item.ctx, m);
            m = nullptr;
        }
    }
    ctx_item.output_mems.clear();
    ctx_item.io_output_attrs.clear();
    ctx_item.use_io_mem = false;
}

void YOLOV5_RKNN::start_workers() {
    if (workers_started_) {
        return;
    }
    stop_workers_ = false;
    for (size_t i = 0; i < kRknnContextCount; ++i) {
        workers_[i] = std::thread([this, i] { worker_loop(i); });
    }
    workers_started_ = true;
}

void YOLOV5_RKNN::stop_workers() {
    if (!workers_started_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t: workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_started_ = false;
}

void YOLOV5_RKNN::worker_loop(size_t ctx_index) {
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

        const auto t0 = std::chrono::steady_clock::now();
        if (!infer(job.input_rgb, ctx_index)) {
            release_outputs(ctx_index);
            result.ok       = false;
            result.infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - t0
            )
                                  .count();
            job.promise.set_value(std::move(result));
            continue;
        }

        const auto t1 = std::chrono::steady_clock::now();

        result = decode_outputs(ctx_index);
        release_outputs(ctx_index);

        result.infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        job.promise.set_value(std::move(result));
    }
}

bool YOLOV5_RKNN::preprocess(const cv::Mat& raw_img, cv::Mat& input_rgb, double& scale) const {
    if (raw_img.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return false;
    }

    cv::Mat bgr_img;
    if (use_roi_) {
        cv::Rect roi = roi_;
        if (roi.width == -1) { // -1 表示该维度不裁切
            roi.width = raw_img.cols;
        }
        if (roi.height == -1) { // -1 表示该维度不裁切
            roi.height = raw_img.rows;
        }
        if (roi.width <= 0 || roi.height <= 0) {
            tools::logger()->error("Invalid ROI size: {}x{}", roi.width, roi.height);
            return false;
        }
        bgr_img = raw_img(roi);
    } else {
        bgr_img = raw_img;
    }

    if (bgr_img.empty() || bgr_img.rows <= 0 || bgr_img.cols <= 0) {
        tools::logger()->error("Empty ROI image");
        return false;
    }

    auto h_scale = static_cast<double>(kInputSize) / bgr_img.rows;
    auto w_scale = static_cast<double>(kInputSize) / bgr_img.cols;
    scale        = std::min(h_scale, w_scale);
    auto h       = static_cast<int>(bgr_img.rows * scale);
    auto w       = static_cast<int>(bgr_img.cols * scale);
    h            = std::max(h, 1);
    w            = std::max(w, 1);

    // preprocess (letterbox)
    auto input = cv::Mat(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    auto roi   = cv::Rect(0, 0, w, h);
    cv::resize(bgr_img, input(roi), { w, h });

    cv::cvtColor(input, input_rgb, cv::COLOR_BGR2RGB);
    return true;
}

std::future<YOLOV5_RKNN::InferResult> YOLOV5_RKNN::enqueue_infer(const cv::Mat& input_rgb) {
    InferJob job;
    job.input_rgb = input_rgb;
    auto fut      = job.promise.get_future();

    if (workers_started_) {
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            job_queue_.push_back(std::move(job));
        }
        queue_cv_.notify_one();
        return fut;
    }

    // Fallback: no workers, run inline
    const auto t0 = std::chrono::steady_clock::now();
    const size_t ctx_index =
        static_cast<size_t>(next_ctx_.fetch_add(1, std::memory_order_relaxed) % kRknnContextCount);

    InferResult result;
    result.ctx_index = ctx_index;

    if (!infer(input_rgb, ctx_index)) {
        release_outputs(ctx_index);
        result.ok       = false;
        result.infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0
        )
                              .count();
        job.promise.set_value(std::move(result));
        return fut;
    }

    const auto t1 = std::chrono::steady_clock::now();

    result = decode_outputs(ctx_index);
    release_outputs(ctx_index);

    result.infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    job.promise.set_value(std::move(result));
    return fut;
}

bool YOLOV5_RKNN::infer(const cv::Mat& img_rgb_u8, size_t ctx_index) {
    if (ctx_index >= kRknnContextCount || ctxs_[ctx_index].ctx == 0) {
        tools::logger()->error("RKNN context is null");
        return false;
    }
    if (img_rgb_u8.empty() || img_rgb_u8.type() != CV_8UC3) {
        tools::logger()->error("Expect RGB uint8 HWC CV_8UC3 input");
        return false;
    }

    auto& ctx_item = ctxs_[ctx_index];

    if (ctx_item.use_io_mem) {
        // Zero-copy path: copy image data directly to DMA buffer
        if (!ctx_item.input_mem || !ctx_item.input_mem->virt_addr) {
            tools::logger()->error("input_mem is null");
            return false;
        }

        // Determine expected dimensions from input attr
        int req_h = 0, req_w = 0, req_c = 0;
        if (ctx_item.io_input_attr.n_dims >= 4) {
            req_h = static_cast<int>(ctx_item.io_input_attr.dims[1]);
            req_w = static_cast<int>(ctx_item.io_input_attr.dims[2]);
            req_c = static_cast<int>(ctx_item.io_input_attr.dims[3]);
        } else {
            tools::logger()->error("Unexpected input dims");
            return false;
        }

        if (img_rgb_u8.rows != req_h || img_rgb_u8.cols != req_w || img_rgb_u8.channels() != req_c)
        {
            tools::logger()->error(
                "Input image shape mismatch, expect {}x{}x{}, got {}x{}x{}",
                req_w,
                req_h,
                req_c,
                img_rgb_u8.cols,
                img_rgb_u8.rows,
                img_rgb_u8.channels()
            );
            return false;
        }

        // Handle w_stride alignment (DMA buffer may have wider rows)
        const int width    = req_w;
        const int height   = req_h;
        const int channel  = req_c;
        const int w_stride = ctx_item.io_input_attr.w_stride > 0
            ? static_cast<int>(ctx_item.io_input_attr.w_stride)
            : width;

        auto* dst       = reinterpret_cast<uint8_t*>(ctx_item.input_mem->virt_addr);
        const auto* src = img_rgb_u8.ptr<uint8_t>();

        if (w_stride == width) {
            std::memcpy(dst, src, static_cast<size_t>(width) * height * channel);
        } else {
            const int src_wc = width * channel;
            const int dst_wc = w_stride * channel;
            for (int h = 0; h < height; ++h) {
                std::memcpy(
                    dst + static_cast<size_t>(h) * dst_wc,
                    src + static_cast<size_t>(h) * src_wc,
                    static_cast<size_t>(src_wc)
                );
            }
        }

        int ret = rknn_run(ctx_item.ctx, nullptr);
        if (ret != RKNN_SUCC) {
            tools::logger()->error("rknn_run failed, ret={}", ret);
            return false;
        }
        return true;
    }

    // Legacy path: use rknn_inputs_set + rknn_outputs_get
    rknn_input in;
    std::memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type  = RKNN_TENSOR_UINT8;
    in.fmt   = RKNN_TENSOR_NHWC;
    in.size  = static_cast<uint32_t>(img_rgb_u8.total() * img_rgb_u8.elemSize());
    in.buf   = const_cast<unsigned char*>(img_rgb_u8.ptr<unsigned char>());

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

    auto& fallback = g_fallback_outputs[ctx_index];
    fallback.assign(ctx_item.io_num.n_output, {});
    for (uint32_t i = 0; i < ctx_item.io_num.n_output; ++i) {
        fallback[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_item.ctx, ctx_item.io_num.n_output, fallback.data(), nullptr);
    if (ret != RKNN_SUCC) {
        tools::logger()->error("rknn_outputs_get failed, ret={}", ret);
        fallback.clear();
        return false;
    }

    return true;
}

YOLOV5_RKNN::InferResult YOLOV5_RKNN::decode_outputs(size_t ctx_index) {
    InferResult result;
    result.ctx_index = ctx_index;

    if (ctx_index >= kRknnContextCount) {
        result.ok = false;
        return result;
    }

    const auto& ctx_item     = ctxs_[ctx_index];
    const auto& output_attrs = ctx_item.output_attrs;
    const uint32_t n_output  = ctx_item.io_num.n_output;

    // Build pointer array to output data
    std::vector<const float*> out_ptrs(n_output, nullptr);

    if (ctx_item.use_io_mem) {
        for (uint32_t i = 0; i < n_output; ++i) {
            if (i < ctx_item.output_mems.size() && ctx_item.output_mems[i]) {
                out_ptrs[i] = reinterpret_cast<const float*>(ctx_item.output_mems[i]->virt_addr);
            }
        }
    } else {
        const auto& fallback = g_fallback_outputs[ctx_index];
        for (uint32_t i = 0; i < n_output && i < fallback.size(); ++i) {
            out_ptrs[i] = reinterpret_cast<const float*>(fallback[i].buf);
        }
    }

    // Decode 3-branch outputs into [25200, 22]
    if (n_output >= static_cast<uint32_t>(kNumScales) && out_ptrs[0] && out_ptrs[1] && out_ptrs[2])
    {
        std::array<const float*, 3> ptrs = { out_ptrs[0], out_ptrs[1], out_ptrs[2] };
        auto decoded                     = decode_3branch(ptrs, output_attrs);
        const int total                  = static_cast<int>(decoded.size()) / kOutputCols;
        result.ok                        = true;
        result.rows                      = total;
        result.cols                      = kOutputCols;
        result.output                    = std::move(decoded);
    } else if (n_output >= 1 && out_ptrs[0]) {
        // Fallback for single-head model (legacy path)
        int rows = 0, cols = 0;
        if (!output_attrs.empty()) {
            const auto& out_attr = output_attrs.front();
            if (out_attr.n_dims >= 3) {
                rows = static_cast<int>(out_attr.dims[1]);
                cols = static_cast<int>(out_attr.dims[2]);
            }
        }
        if (rows <= 0 || cols <= 0) {
            rows = kOutputRowsFallback;
            cols = kOutputCols;
        }
        result.ok   = true;
        result.rows = rows;
        result.cols = cols;
        result.output.assign(out_ptrs[0], out_ptrs[0] + static_cast<size_t>(rows) * cols);
    } else {
        result.ok = false;
    }

    return result;
}

void YOLOV5_RKNN::release_outputs(size_t ctx_index) {
    if (ctx_index >= kRknnContextCount || ctxs_[ctx_index].ctx == 0) {
        return;
    }
    if (ctxs_[ctx_index].use_io_mem) {
        // Zero-copy: output buffers persist, nothing to release
        return;
    }
    auto& fallback = g_fallback_outputs[ctx_index];
    if (!fallback.empty()) {
        rknn_outputs_release(ctxs_[ctx_index].ctx, fallback.size(), fallback.data());
        fallback.clear();
    }
}

bool YOLOV5_RKNN::read_file(const std::string& path, std::vector<uint8_t>& data) {
    FILE* fp = fopen(path.c_str(), "rb");
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

} // namespace auto_aim
