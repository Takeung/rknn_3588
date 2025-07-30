// Filename: src/video_publisher.cpp

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class VideoPublisher : public rclcpp::Node {
  public:
    VideoPublisher()
        : Node("video_publisher"),
          cap_("/home/rknn/dev_ws_det2d/src/dev_pkg_det2d/media/bj_short.mp4") {
        publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>("video_frames", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(
                static_cast<int>(1000 / cap_.get(cv::CAP_PROP_FPS))),
            std::bind(&VideoPublisher::timer_callback, this));
    }

  private:
    void timer_callback() {
        cv::Mat frame;
        if (cap_.read(frame)) {
            auto msg =
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame)
                    .toImageMsg();
            msg->header.stamp = this->get_clock()->now();
            msg->header.frame_id = "camera_frame";
            publisher_->publish(*msg);
        } else {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);  // Loop the video
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());
    rclcpp::shutdown();
    return 0;
}