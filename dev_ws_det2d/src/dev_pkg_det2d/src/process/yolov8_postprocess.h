#ifndef _RKNN_YOLOV8_POSTPROCESS_H_
#define _RKNN_YOLOV8_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <string>
// #include "rknn_api.h"
#include "types/datatype.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

namespace yolov8 {

// typedef struct {
//     rknn_context rknn_ctx;
//     rknn_input_output_num io_num;
//     rknn_tensor_attr* input_attrs;
//     rknn_tensor_attr* output_attrs;
//     int model_channel;
//     int model_width;
//     int model_height;
//     bool is_quant;
// } rknn_app_context_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    char name[OBJ_NAME_MAX_SIZE];
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int post_process(std::vector<tensor_data_s>& output_tensors, int model_in_h,
                 int model_in_w, float conf_threshold, float nms_threshold,
                 float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales,
                 object_detect_result_list* od_results);

bool load_parameters_from_json(const std::string& json_file);

extern int obj_class_num_;
extern int all_class_num_;
extern int prop_box_size_;
extern std::vector<std::string> labels_;

}  // namespace yolov8

#endif  //_RKNN_POSTPROCESS_H_
