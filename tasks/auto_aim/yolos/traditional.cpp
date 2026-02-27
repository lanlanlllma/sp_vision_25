#include "traditional.hpp"

#include "tools/logger.hpp"

namespace auto_aim {
Traditional::Traditional(const std::string& config_path, bool debug):
    detector_(config_path, debug) {}

std::list<Armor> Traditional::detect(const cv::Mat& bgr_img, int frame_count) {
    if (bgr_img.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return std::list<Armor>();
    }

    return detector_.detect(bgr_img, frame_count);
}

std::list<Armor>
Traditional::postprocess(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count) {
    // 传统方法不使用神经网络，postprocess 直接调用 detect
    return detector_.detect(bgr_img, frame_count);
}

} // namespace auto_aim
