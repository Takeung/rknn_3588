#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <string>

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128

namespace yolov5 {

typedef struct _BOX_RECT {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    int id;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t {
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h,
                 int model_in_w, float conf_threshold, float nms_threshold,
                 float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group);

bool load_parameters_from_json(const std::string& json_file);

extern int obj_class_num_;
extern int all_class_num_;
extern int prop_box_size_;
extern std::vector<std::string> labels_;
extern int anchor0_[6];
extern int anchor1_[6];
extern int anchor2_[6];

}  // namespace yolov5

#endif  //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
