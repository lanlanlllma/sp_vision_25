#include "cboard.hpp"

#include "Eigen/src/Core/AssignEvaluator.h"

#include <cstdint>
#include <iostream>

namespace io {
CBoard::CBoard(const std::string& config_path):
    mode(Mode::idle),
    shoot_mode(ShootMode::left_shoot),
    bullet_speed(21.0),
    queue_(5000) {
    auto yaml          = YAML::LoadFile(config_path);
    this->bullet_speed = yaml["bullet_speed"].as<double>();

    tools::logger()->info("[Cboard] Waiting for q...");

    this->read_buffer_.resize(32);
    this->write_buffer_.resize(32);
    this->serial_ = serial_phoenix::Serial();

    auto code = this->serial_.open(findFirstACMDevice(), nullptr, 32);
    if (!code) {
        tools::logger()->warn("[Cboard] Serial port not opened: {}", static_cast<int>(code.code()));
    }
    this->start();
    queue_.pop(data_ahead_);
    queue_.pop(data_behind_);
    tools::logger()->info("[Cboard] Opened.");
}

void CBoard::start() {
    std::thread Link_thread([this] {
        while (true) {
            this->serial_.read(this->read_buffer_);
            // std::cout << "Data received from serial port." << std::endl;
            // for (auto it = this->read_buffer_.begin(); it != this->read_buffer_.end(); ++it) {
            //     std::cout << std::hex << static_cast<int>(*it) << " ";
            // }
            // std::cout << std::dec;
            // std::cout << std::endl;
            // 解析数据
            uint8_t type = this->read_buffer_[1];
            // std::cout << "a." << std::endl;

            std::vector<uint8_t> buffer(29);

            std::memcpy(buffer.data(), this->read_buffer_.data() + 2, 29);
            if (type == 0xb0) {
                this->read_fun_1(*(Message_phoenix*)this->read_buffer_.data());
                // std::cout << "Received IMU data." << std::endl;
            } else {
                std::cout << "Unknown message type: " << std::hex << static_cast<int>(type)
                          << std::dec << std::endl;
            }
        }
    });
    Link_thread.detach();
}

Eigen::Quaterniond CBoard::imu_at(std::chrono::steady_clock::time_point timestamp) {
    if (data_behind_.timestamp < timestamp)
        data_ahead_ = data_behind_;

    while (true) {
        queue_.pop(data_behind_);
        if (data_behind_.timestamp > timestamp)
            break;
        data_ahead_ = data_behind_;
    }

    Eigen::Quaterniond q_a             = data_ahead_.q.normalized();
    Eigen::Quaterniond q_b             = data_behind_.q.normalized();
    auto t_a                           = data_ahead_.timestamp;
    auto t_b                           = data_behind_.timestamp;
    auto t_c                           = timestamp;
    std::chrono::duration<double> t_ab = t_b - t_a;
    std::chrono::duration<double> t_ac = t_c - t_a;

    // 四元数插值
    auto k                 = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();

    return q_c;
}

// 存在多写竞争，但是目前只有单线程写入需求
// TODO: 改进为线程安全
void CBoard::send(Command command) {
    Message_phoenix msg;
    msg.header = 's';
    msg.type   = 0xA0;
    GimbalControl msg_body;
    msg_body.find_bools = command.control ? 49 : 48; // '1' or '0'
    msg_body.yaw        = -(static_cast<float>(command.yaw)+3.14159);
    //into 0-2pi
    if (msg_body.yaw < 0) {
        msg_body.yaw += 2 * 3.14159;
    }
    msg_body.pitch      = (static_cast<float>(command.pitch)+3.14159/2);
    //into 0-2pi
    if (msg_body.pitch < 0) {
        msg_body.pitch += 2 * 3.14159;
    }
    std::memcpy(msg.data, &msg_body, sizeof(GimbalControl));
    msg.tail = 'e';

    auto code = this->serial_.write(std::move(msg));
    if (!code) {
        tools::logger()->warn("Serial write failed: {}", static_cast<int>(code.code()));
    }
}

// 串口通信下已弃用
// std::string CBoard::read_yaml(const std::string& config_path) {
//     auto yaml = tools::load(config_path);

//     quaternion_canid_   = tools::read<int>(yaml, "quaternion_canid");
//     bullet_speed_canid_ = tools::read<int>(yaml, "bullet_speed_canid");
//     send_canid_         = tools::read<int>(yaml, "send_canid");

//     if (!yaml["can_interface"]) {
//         throw std::runtime_error("Missing 'can_interface' in YAML configuration.");
//     }

//     return yaml["can_interface"].as<std::string>();
// }

void CBoard::read_fun_1(Message_phoenix& msg) {
    auto timestamp = std::chrono::steady_clock::now();

    Autoaim_s data = reinterpret_cast<Autoaim_s&>(msg.data);
    float yaw      = (-data.yaw+(3.14159));
    // into 0-2pi
    if (yaw < 0) {
        yaw += 2 * 3.14159;
    }
    float pitch    = (data.pitch)-3.14159/2;
    if (pitch < 0) {
        pitch += 2 * 3.14159;
    }

    // std::cout << "Yaw: " << yaw << ", Pitch: " << pitch << std::endl;
    Eigen::AngleAxisd yaw_aa(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitch_aa(pitch, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond q = (yaw_aa * pitch_aa).normalized();
    queue_.push({ q, timestamp });
}

std::string CBoard::findFirstACMDevice() {
    const std::string dev_path = "/dev/";
    for (const auto& entry: std::filesystem::directory_iterator(dev_path)) {
        if (entry.path().filename().string().find("ttyACM") != std::string::npos) {
            std::cout << "find ACM device: " << entry.path().string() << std::endl;
            return entry.path().string();
        }
    }
    throw std::runtime_error("No /dev/ttyACM* device found.");
}
} // namespace io