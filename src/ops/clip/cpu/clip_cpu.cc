#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
    ClipCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y
    ){

    if (x->dt != F16 && x->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (x->ndim != y->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(x) || !is_contiguous(y)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    uint64_t element_num = 1;
    for (uint64_t i = 0; i < x->ndim; i++) {
        element_num *= x->shape[i];
    }
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        x->dt,
        element_num
    };
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc,
    void const *x,
    void *min,
    void *max,
    void *y){
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    float min_val;
    float max_val;
    auto min_ = reinterpret_cast<float *>(min);
    min_val = *min_;
    auto max_ = reinterpret_cast<float *>(max);
    max_val = *max_;

    for (uint64_t i = 0; i < desc->element_num; i++) {
        if constexpr (std::is_same<Tdata, uint16_t>::value){
            y_[i] = f32_to_f16(std::min(std::max(f16_to_f32(x_[i]), min_val), max_val));
        }
        else{
            y_[i] = std::min(std::max(x_[i], min_val), max_val);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
    void const*x,
    void *min,
    void *max,
    void *y,
    void *stream){
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, x, min, max, y);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, x, min, max, y);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc){
    delete desc;
    return STATUS_SUCCESS;
}