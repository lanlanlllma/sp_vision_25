#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明 }"
    "{@config-path   | configs/sentry.yaml    | yaml配置文件的路径}"
    "{tradition t    |  false                 | 是否使用传统方法识别}";

int main(int argc, char* argv[]) {
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto config_path   = cli.get<std::string>(0);
    auto use_tradition = cli.get<bool>("tradition");

    tools::Exiter exiter;

    io::Camera camera(config_path);
    auto_aim::Detector detector(config_path, true);
    auto_aim::YOLO yolo(config_path, true);

    std::chrono::steady_clock::time_point timestamp;

    auto overall_start = std::chrono::steady_clock::now();
    size_t frame_count = 0;

    while (!exiter.exit()) {
        cv::Mat img;
        std::list<auto_aim::Armor> armors;

        auto loop_start = std::chrono::steady_clock::now();

        auto read_start = std::chrono::steady_clock::now();
        camera.read(img, timestamp);
        auto read_end = std::chrono::steady_clock::now();

        if (img.empty())
            break;

        ++frame_count;

        auto detect_start = std::chrono::steady_clock::now();

        if (use_tradition)
            armors = detector.detect(img);
        else
            armors = yolo.detect(img);

        auto detect_end = std::chrono::steady_clock::now();
        auto loop_end   = std::chrono::steady_clock::now();

        auto read_dt       = tools::delta_time(read_end, read_start);
        auto detect_dt     = tools::delta_time(detect_end, detect_start);
        auto loop_dt       = tools::delta_time(loop_end, loop_start);
        auto elapsed_s     = tools::delta_time(loop_end, overall_start);
        double overall_fps = 0.0;
        if (elapsed_s > 0.0)
            overall_fps = static_cast<double>(frame_count) / elapsed_s;

        tools::logger()->info(
            "[read] {:.2f} ms | [detect] {:.2f} ms | [loop] {:.2f} ms | [overall_fps] {:.2f}",
            read_dt * 1000,
            detect_dt * 1000,
            loop_dt * 1000,
            overall_fps
        );

        auto key = cv::waitKey(33);
        if (key == 'q')
            break;
    }

    return 0;
}