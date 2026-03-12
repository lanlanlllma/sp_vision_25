#include <atomic>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                  | 输出命令行参数说明}"
  "{@config-path   | configs/uav.yaml | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::CBoard cboard(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  std::atomic<io::Mode> mode{io::Mode::idle};
  io::Mode last_mode = io::Mode::idle;

  std::mutex camera_mtx;

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      const auto m = mode.load();
      if (m == io::Mode::auto_aim || m == io::Mode::outpost) {
        {
          std::scoped_lock<std::mutex> lk(camera_mtx);
          camera.read(img, t);
        }
        if (!img.empty()) detector.push(img, t);
      } else {
        std::this_thread::sleep_for(1ms);
      }
    }
  });

  while (!exiter.exit()) {
    const io::Mode current_mode = cboard.mode;
    mode.store(current_mode);

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", io::MODES[current_mode]);
      last_mode = current_mode;
    }

    /// 自瞄/前哨站
    if (current_mode == io::Mode::auto_aim || current_mode == io::Mode::outpost) {
      auto [img, armors, t] = detector.debug_pop();
      const Eigen::Quaterniond q = cboard.imu_at(t - 1ms);
      // recorder.record(img, q, t);

      solver.set_R_gimbal2world(q);
      const Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto targets = tracker.track(armors, t);
      commandgener.push(targets, t, cboard.bullet_speed, ypr);
    }

    /// 打符
    else if (current_mode == io::Mode::small_buff || current_mode == io::Mode::big_buff) {
      cv::Mat img;
      std::chrono::steady_clock::time_point t;
      {
        std::scoped_lock<std::mutex> lk(camera_mtx);
        camera.read(img, t);
      }
      if (img.empty()) continue;

      const Eigen::Quaterniond q = cboard.imu_at(t - 1ms);
      // recorder.record(img, q, t);

      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);
      buff_solver.solve(power_runes);

      io::Command buff_command;
      if (current_mode == io::Mode::small_buff) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_command = buff_aimer.aim(target_copy, t, cboard.bullet_speed, true);
      } else {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_command = buff_aimer.aim(target_copy, t, cboard.bullet_speed, true);
      }
      cboard.send(buff_command);
    }

    else {
      std::this_thread::sleep_for(1ms);
    }
  }

  if (detect_thread.joinable()) detect_thread.join();
  return 0;
}
