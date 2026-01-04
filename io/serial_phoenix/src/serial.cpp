#include "../include/serial.hpp"
#include <cstring>

namespace serial_phoenix {

Serial::Serial(Serial&& other) noexcept {
#ifdef SP_VISION_HAVE_SERIAL_DRIVER
    owned_ctx_   = std::move(other.owned_ctx_);
    serial_port_ = std::move(other.serial_port_);
#else
    serial_raw_ = std::move(other.serial_raw_);
#endif
    read_buffer_  = std::move(other.read_buffer_);
    write_buffer_ = std::move(other.write_buffer_);
    bytes_        = other.bytes_;
}

Serial& Serial::operator=(Serial&& other) noexcept {
    if (this != &other) {
#ifdef SP_VISION_HAVE_SERIAL_DRIVER
        owned_ctx_   = std::move(other.owned_ctx_);
        serial_port_ = std::move(other.serial_port_);
#else
        serial_raw_ = std::move(other.serial_raw_);
#endif
        read_buffer_  = std::move(other.read_buffer_);
        write_buffer_ = std::move(other.write_buffer_);
        bytes_        = other.bytes_;
    }
    return *this;
}

SerialCode Serial::open(std::string port, std::shared_ptr<SPconfig> config, size_t bytes_) {
    if (this->is_open()) {
        return SerialCode::Value::OPEN_OPENED;
    }
    if (!std::filesystem::exists(port)) {
        return SerialCode::Value::OPEN_DEV_NOT_EXIST;
    }

    this->bytes_ = bytes_;
    this->read_buffer_.resize(this->bytes_);
    this->write_buffer_.resize(this->bytes_);

#ifdef SP_VISION_HAVE_SERIAL_DRIVER
    if (config == nullptr) {
        config = std::make_shared<SPconfig>(
            uint32_t(115200),
            drivers::serial_driver::FlowControl::NONE,
            drivers::serial_driver::Parity::NONE,
            drivers::serial_driver::StopBits::ONE
        );
    }
    this->owned_ctx_ = std::make_unique<IoContext>(2);
    this->serial_port_ =
        std::make_shared<drivers::serial_driver::SerialPort>(*owned_ctx_, port, *config);
    this->serial_port_->open();

    if (this->serial_port_->is_open()) {
        return SerialCode::Value::OK;
    } else {
        return SerialCode::Value::OPEN_FAIL;
    }
#else
    try {
        // 使用纯 C++ 串口库，配置为常用的 115200 波特率
        serial_raw_ =
            std::make_unique<serial::Serial>(port, 115200, serial::Timeout::simpleTimeout(100));
    } catch (const std::exception&) {
        return SerialCode::Value::OPEN_FAIL;
    }

    return (serial_raw_ && serial_raw_->isOpen()) ? SerialCode::Value::OK
                                                  : SerialCode::Value::OPEN_FAIL;
#endif
}

bool Serial::is_open() const {
#ifdef SP_VISION_HAVE_SERIAL_DRIVER
    return this->serial_port_ && this->serial_port_->is_open();
#else
    return this->serial_raw_ && this->serial_raw_->isOpen();
#endif
}

bool Serial::SetByte(size_t bytes) {
    if (this->is_open()) {
        this->bytes_ = bytes;
        return true;
    }
    return false;
}

SerialCode Serial::close() {
    if (!this->is_open()) {
        return SerialCode::Value::CLOSE_NOT_OPENED;
    }

#ifdef SP_VISION_HAVE_SERIAL_DRIVER
    if (this->serial_port_ && this->serial_port_->is_open()) {
        this->serial_port_->close();
        this->owned_ctx_.reset();
    }
    this->bytes_ = 0;
    return this->serial_port_ && this->serial_port_->is_open() ? SerialCode::Value::CLOSE_FAIL
                                                               : SerialCode::Value::OK;
#else
    if (this->serial_raw_ && this->serial_raw_->isOpen()) {
        try {
            this->serial_raw_->close();
        } catch (const std::exception&) {
            return SerialCode::Value::CLOSE_FAIL;
        }
    }
    this->bytes_ = 0;
    return this->is_open() ? SerialCode::Value::CLOSE_FAIL : SerialCode::Value::OK;
#endif
}

} // namespace serial_phoenix