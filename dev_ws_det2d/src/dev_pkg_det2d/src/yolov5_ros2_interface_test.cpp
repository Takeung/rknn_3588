
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "draw/cv_draw.h"
#include "task/yolov5.h"
#include "utils/logging.h"

#include "task/yolov5_thread_pool.h"

#include "dev_pkg_interfaces/msg/bbox.hpp"
#include "dev_pkg_interfaces/msg/bbox_array.hpp"

class Det2dNode : public rclcpp::Node {
  public:
    Det2dNode(std::string& model_file, const int num_threads,
              std::string& input_topic_name, std::string& output_obj_topic_name,
              std::string& output_img_topic_name)
        : Node("det2d_node"),
          last_message_time_(std::chrono::steady_clock::now()),
          output_obj_topic_name_(output_obj_topic_name),
          output_img_topic_name_(output_img_topic_name) {
        // rknn infer
        g_pool = new Yolov5ThreadPool();
        g_pool->setUp(model_file, num_threads);
        // read stream
        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic_name, 10,
            std::bind(&Det2dNode::read_stream_callback, this,
                      std::placeholders::_1));
        // get results
        obj_publisher_ =
            this->create_publisher<dev_pkg_interfaces::msg::BboxArray>(
                output_obj_topic_name, 10);
        img_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            output_img_topic_name, 10);
        get_results_thread_ =
            std::thread(&Det2dNode::get_results_callback, this);
        // monitor
        timer_ =
            this->create_wall_timer(std::chrono::seconds(1),
                                    std::bind(&Det2dNode::check_timeout, this));
    }
    ~Det2dNode() { stop(); }

    void start() { rclcpp::spin(this->get_node_base_interface()); }
    void stop() {
        rclcpp::shutdown();
        g_pool->stopAll();
        if (get_results_thread_.joinable()) {
            get_results_thread_.join();
        }
    }

  private:
    void read_stream_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        last_message_time_ = std::chrono::steady_clock::now();

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr =
                cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s",
                         e.what());
            return;
        }
        // // debug: show input
        // cv::imshow("Video Frame", cv_ptr->image);
        // cv::waitKey(1);
        g_pool->submitTask(cv_ptr->image.clone(), g_frame_start_id++);
    }

    void get_results_callback() {
        // 记录开始时间
        auto start_all = std::chrono::high_resolution_clock::now();
        int frame_count = 0;

        while (rclcpp::ok()) {
            if (output_obj_topic_name_ != "null") {
                std::vector<Detection> objects;
                auto ret = g_pool->getTargetResult(objects, g_frame_end_id);
                if (ret != NN_SUCCESS) {
                    NN_LOG_WARNING("Failed to get results, Status:%d", ret);
                    continue;
                }

                // publish bbbox
                if (!objects.empty()) {
                    auto message_bboxes = dev_pkg_interfaces::msg::BboxArray();
                    message_bboxes.frame_id = g_frame_end_id;
                    for (auto& object : objects) {
                        auto message_bbox = dev_pkg_interfaces::msg::Bbox();
                        message_bbox.class_id = object.class_id;
                        message_bbox.x_left = object.box.x;
                        message_bbox.y_top = object.box.y;
                        message_bbox.width = object.box.width;
                        message_bbox.height = object.box.height;
                        message_bboxes.boxes.push_back(message_bbox);
                    }
                    obj_publisher_->publish(message_bboxes);
                }
            }

            if (output_img_topic_name_ != "null") {
                cv::Mat img;
                auto ret = g_pool->getTargetImgResult(img, g_frame_end_id);
                if (ret != NN_SUCCESS) {
                    NN_LOG_WARNING("Failed to get results, Status:%d", ret);
                    continue;
                }

                // publish img
                if (!img.empty()) {
                    auto msg =
                        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                            .toImageMsg();
                    msg->header.stamp = this->get_clock()->now();
                    msg->header.frame_id = g_frame_end_id;
                    img_publisher_->publish(*msg);
                }
            }

            g_frame_end_id++;

            // 算法2：计算超过 1s 一共处理了多少张图片
            frame_count++;
            // all end
            auto end_all = std::chrono::high_resolution_clock::now();
            auto elapsed_all_2 =
                std::chrono::duration_cast<std::chrono::microseconds>(end_all -
                                                                      start_all)
                    .count() /
                1000.f;
            // 每隔1秒打印一次
            if (elapsed_all_2 > 1000) {
                NN_LOG_INFO(
                    "Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2,
                    frame_count / (elapsed_all_2 / 1000.0f), frame_count);
                frame_count = 0;
                start_all = std::chrono::high_resolution_clock::now();
            }
        }
    }

    void check_timeout() {
        auto now = std::chrono::steady_clock::now();
        auto duration_since_last_message =
            std::chrono::duration_cast<std::chrono::seconds>(
                now - last_message_time_);
        if (duration_since_last_message.count() > 5) {
            RCLCPP_WARN(this->get_logger(),
                        "No messages received for 5 seconds.");
            // this->stop();
        }
    }

  private:
    int g_frame_start_id = 0;  // 读取视频帧的索引
    int g_frame_end_id = 0;    // 模型处理完的索引
    Yolov5ThreadPool* g_pool = nullptr;
    // std::string input_topic_name_;
    std::string output_obj_topic_name_;
    std::string output_img_topic_name_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Publisher<dev_pkg_interfaces::msg::BboxArray>::SharedPtr
        obj_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_publisher_;
    std::thread get_results_thread_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::chrono::steady_clock::time_point last_message_time_;
};

// Signal handler function
void signal_handler(int signum) {
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    // Register signal handler
    signal(SIGINT, signal_handler);

    // model file path
    std::string model_file = argv[1];
    // input video topic
    std::string video_topic_name = argv[2];
    // 参数：线程池数量
    const int num_threads = (argc > 3) ? atoi(argv[3]) : 12;
    // output objects topic
    std::string objects_topic_name = (argc > 4) ? argv[4] : "";
    // output images topic
    std::string images_topic_name = (argc > 5) ? argv[5] : "";

    rclcpp::init(argc, argv);
    auto det2d_node =
        std::make_shared<Det2dNode>(model_file, num_threads, video_topic_name,
                                    objects_topic_name, images_topic_name);
    det2d_node->start();
    det2d_node->stop();

    return 0;
}