#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <unordered_set>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径}";

tools::ThreadSafeQueue<tools::Frame, true> frame_queue{128};

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  const int num_yolo_thread = 3;
  std::vector<auto_aim::YOLO> yolos;
  yolos.reserve(num_yolo_thread);
  for (int i = 0; i < num_yolo_thread; ++i) {
    yolos.emplace_back(config_path);
  }
  auto_aim::Solver solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  auto t0 = std::chrono::steady_clock::now();

  auto_aim::Target last_target;
  io::Command last_command;
  double last_t = -1;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  tools::ThreadPool thread_pool(num_yolo_thread);
  std::queue<int> yolo_free;
  std::mutex yolo_mu;
  std::condition_variable yolo_cv;
  for (int i = 0; i < num_yolo_thread; ++i) {
    yolo_free.push(i);
  }

  std::atomic<bool> stop_requested{false};
  std::atomic<bool> done_reading{false};
  std::atomic<int> processed_frames{0};
  const auto overall_start = std::chrono::steady_clock::now();

  auto process_thread = std::thread([&]() {
    struct LightFrame
    {
      int id = 0;
      std::chrono::steady_clock::time_point t;
      Eigen::Quaterniond q;
      std::list<auto_aim::Armor> armors;
    };

    std::map<int, LightFrame> pending;
    std::map<int, LightFrame> history;
    std::map<int, cv::Mat> img_cache;
    std::unordered_set<int> counted_ids;
    int expected_id = 1;
    constexpr size_t kHistoryLimit = 200;
    constexpr size_t kImgCacheLimit = 50;

    auto reset_and_replay = [&]() {
      tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
      last_t = -1;
      pending.clear();
      for (const auto & kv : history) {
        if (kv.first < expected_id) continue;
        pending.emplace(kv.first, kv.second);
      }
    };

    auto process_one = [&](const LightFrame & frame, cv::Mat img) {
      auto timestamp = frame.t;
      const auto t = std::chrono::duration<double>(timestamp - t0).count();
      const auto & gimbal_q = frame.q;

      solver.set_R_gimbal2world(gimbal_q);

      auto tracker_start = std::chrono::steady_clock::now();
      auto armors = frame.armors;
      auto targets = tracker->track(armors, timestamp);

      auto aimer_start = std::chrono::steady_clock::now();
      auto command = aimer.aim(targets, timestamp, 27, false);

      if (
        !targets.empty() && aimer.debug_aim_point.valid &&
        std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
        command.shoot = true;

      if (command.control) last_command = command;

      auto finish = std::chrono::steady_clock::now();
      tools::logger()->info(
        "[{}] tracker: {:.1f}ms, aimer: {:.1f}ms", frame.id,
        tools::delta_time(aimer_start, tracker_start) * 1e3,
        tools::delta_time(finish, aimer_start) * 1e3);

      if (counted_ids.insert(frame.id).second) {
        const int done = ++processed_frames;
        if (done % 30 == 0) {
          const auto elapsed_s = std::chrono::duration<double>(finish - overall_start).count();
          if (elapsed_s > 0.0) {
            const double fps = done / elapsed_s;
            tools::logger()->info(
              "[overall] frames={} elapsed={:.2f}s fps={:.2f}", done, elapsed_s, fps);
          }
        }
      }

    //   if (!img.empty()) {
    //     tools::draw_text(
    //       img,
    //       fmt::format(
    //         "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
    //         command.pitch * 57.3, command.shoot),
    //       {10, 60}, {154, 50, 205});

    //     tools::draw_text(
    //       img,
    //       fmt::format(
    //         "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
    //       {10, 90}, {255, 255, 255});
    //   }

      nlohmann::json data;

      data["armor_num"] = armors.size();
      if (!armors.empty()) {
        const auto & armor = armors.front();
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
        data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
        data["armor_center_x"] = armor.center_norm.x;
        data["armor_center_y"] = armor.center_norm.y;
      }

      auto yaw = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0)[0];
      data["gimbal_yaw"] = yaw * 57.3;
      data["cmd_yaw"] = command.yaw * 57.3;
      data["shoot"] = command.shoot;

      bool skip_frame = false;
      if (!targets.empty()) {
        auto target = targets.front();

        if (last_t == -1) {
          last_target = target;
          last_t = t;
          skip_frame = true;
        } else {
          std::vector<Eigen::Vector4d> armor_xyza_list;

          armor_xyza_list = target.armor_xyza_list();
          if (!img.empty()) {
            for (const Eigen::Vector4d & xyza : armor_xyza_list) {
              auto image_points =
                solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
              tools::draw_points(img, image_points, {0, 255, 0});
            }
          }

          auto aim_point = aimer.debug_aim_point;
          Eigen::Vector4d aim_xyza = aim_point.xyza;
          if (!img.empty()) {
            auto image_points =
              solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});
          }

          Eigen::VectorXd x = target.ekf_x();
          data["x"] = x[0];
          data["vx"] = x[1];
          data["y"] = x[2];
          data["vy"] = x[3];
          data["z"] = x[4];
          data["vz"] = x[5];
          data["a"] = x[6] * 57.3;
          data["w"] = x[7];
          data["r"] = x[8];
          data["l"] = x[9];
          data["h"] = x[10];
          data["last_id"] = target.last_id;

          data["residual_yaw"] = target.ekf().data.at("residual_yaw");
          data["residual_pitch"] = target.ekf().data.at("residual_pitch");
          data["residual_distance"] = target.ekf().data.at("residual_distance");
          data["residual_angle"] = target.ekf().data.at("residual_angle");
          data["nis"] = target.ekf().data.at("nis");
          data["nees"] = target.ekf().data.at("nees");
          data["nis_fail"] = target.ekf().data.at("nis_fail");
          data["nees_fail"] = target.ekf().data.at("nees_fail");
          data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
        }
      }

      if (skip_frame) {
        return;
      }

      plotter.plot(data);

      if (!img.empty()) {
        auto key = cv::waitKey(1);
        if (key == 'q') {
          stop_requested = true;
        }
      }
    };

    while (!exiter.exit() && !stop_requested.load()) {
      auto frame = frame_queue.pop();
      if (frame.id < 0) {
        break;
      }

      LightFrame light;
      light.id = frame.id;
      light.t = frame.t;
      light.q = frame.q;
      light.armors = frame.armors;

      pending[frame.id] = light;
      history[frame.id] = light;
      if (history.size() > kHistoryLimit) {
        history.erase(history.begin());
      }

      if (!frame.img.empty()) {
        img_cache[frame.id] = frame.img;
        if (img_cache.size() > kImgCacheLimit) {
          img_cache.erase(img_cache.begin());
        }
      }

      if (frame.id < expected_id) {
        tools::logger()->warn(
          "Out-of-order frame detected: id={} expected={} -> rewind", frame.id, expected_id);
        expected_id = history.begin()->first;
        reset_and_replay();
      }

      while (pending.count(expected_id) > 0) {
        auto it = pending.find(expected_id);
        if (it == pending.end()) break;
        cv::Mat img;
        auto it_img = img_cache.find(expected_id);
        if (it_img != img_cache.end()) {
          img = it_img->second;
          img_cache.erase(it_img);
        }
        process_one(it->second, img);
        pending.erase(it);
        expected_id++;
      }
    }
  });

  int frame_id = 0;
  for (int frame_count = start_index; !exiter.exit() && !stop_requested.load(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    tools::Frame frame;
    frame.id = ++frame_id;
    frame.img = img.clone();
    frame.t = timestamp;
    frame.q = Eigen::Quaterniond(w, x, y, z);

    thread_pool.enqueue([&, frame = std::move(frame)]() mutable {
      int yolo_id = -1;
      {
        std::unique_lock<std::mutex> lk(yolo_mu);
        yolo_cv.wait(lk, [&] {
          return !yolo_free.empty() || exiter.exit() || stop_requested.load();
        });
        if (exiter.exit() || stop_requested.load()) return;
        yolo_id = yolo_free.front();
        yolo_free.pop();
      }

      frame.armors = yolos[yolo_id].detect(frame.img, frame.id);
      frame_queue.push(frame);

      {
        std::lock_guard<std::mutex> lk(yolo_mu);
        yolo_free.push(yolo_id);
      }
      yolo_cv.notify_one();
    });
  }

  done_reading = true;
  frame_queue.push({-1, cv::Mat(), std::chrono::steady_clock::now(), Eigen::Quaterniond::Identity(), {}});
  if (process_thread.joinable()) process_thread.join();

  return 0;
}