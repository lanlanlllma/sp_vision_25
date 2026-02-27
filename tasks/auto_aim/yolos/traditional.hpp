#ifndef AUTO_AIM__TRADITIONAL_HPP
#define AUTO_AIM__TRADITIONAL_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim {
class Traditional: public YOLOBase {
public:
    Traditional(const std::string& config_path, bool debug);

    std::list<Armor> detect(const cv::Mat& bgr_img, int frame_count) override;

    std::list<Armor>
    postprocess(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count) override;

private:
    Detector detector_;
};

} // namespace auto_aim

#endif // AUTO_AIM__TRADITIONAL_HPP
