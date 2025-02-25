#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
    WhereCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src1,
    infiniopTensorDescriptor_t src2,
    infiniopTensorDescriptor_t condition
    ){
    if (dst->dt != F16 && dst->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src1->dt != F16 && src1->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src2->dt != F16 && src2->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(dst) || !is_contiguous(src1) || !is_contiguous(src2) || !is_contiguous(condition)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (dst->dt != src1->dt || dst->dt != src2->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t *src1_shape = new uint64_t[dst->ndim];
    uint64_t *src2_shape = new uint64_t[dst->ndim];
    uint64_t *condition_shape = new uint64_t[dst->ndim];
    uint64_t *dst_shape = new uint64_t[dst->ndim];
    int64_t *dst_strides = new int64_t[dst->ndim];
    uint64_t dst_ndim = dst->ndim;
    uint64_t element_num = 1;
    for (uint64_t i = 0; i < dst->ndim; i++) {
        element_num *= dst->shape[i];
    }
    memcpy(src1_shape, src1->shape, src1->ndim * sizeof(uint64_t));
    memcpy(src2_shape, src2->shape, src2->ndim * sizeof(uint64_t));
    memcpy(condition_shape, condition->shape, condition->ndim * sizeof(uint8_t));
    //扩展到dst的维度
    for (uint64_t i = 0; i < dst->ndim; i++) {
        if (i < src1->ndim){
            src1_shape[i] = src1->shape[i];
        }
        else{
            src1_shape[i] = 1;
        }
        if (i < src2->ndim){
            src2_shape[i] = src2->shape[i];
        }
        else{
            src2_shape[i] = 1;
        }
        if (i < condition->ndim){
            condition_shape[i] = condition->shape[i];
        }
        else{
            condition_shape[i] = 1;
        }
    }
    memcpy(dst_shape, dst->shape, dst->ndim * sizeof(uint64_t));
    memcpy(dst_strides, dst->strides, dst->ndim * sizeof(int64_t));
    *desc_ptr = new WhereCpuDescriptor{
        DevCpu,
        dst->dt,
        src1_shape,
        src2_shape,
        condition_shape,
        dst_shape,
        dst_strides,
        dst_ndim,
        element_num
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc){
    delete[] desc->src1_shape;
    delete[] desc->src2_shape;
    delete[] desc->condition_shape;
    delete[] desc->dst_shape;
    delete[] desc->dst_strides;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc,
    void *dst, 
    void *src1,
    void *src2,
    void *condition,
    void *stream){
    auto dst_ = reinterpret_cast<Tdata *>(dst);
    auto src1_ = reinterpret_cast<Tdata *>(src1);
    auto src2_ = reinterpret_cast<Tdata *>(src2);
    auto condition_ = reinterpret_cast<uint8_t *>(condition);
    auto ndim = desc->dst_ndim;
    auto dst_strides = desc->dst_strides;
    auto condition_shape = desc->condition_shape;
    auto src1_shape = desc->src1_shape;
    auto src2_shape = desc->src2_shape;
    #pragma omp parallel for
    for (uint64_t i = 0; i < desc->element_num; i++) {
        std::vector<int> indices(ndim, 0);
        int global_index = i;
        for (int z = 0; z <= ndim - 1; z++){
            indices[z] = global_index / dst_strides[z];
            global_index %= dst_strides[z];
        }
        uint64_t condition_index = 0, src1_index = 0, src2_index = 0;
        for (int j = 0; j < ndim; j++) {
            if (condition_shape[j] != 1){
                condition_index = condition_index * condition_shape[j] + indices[j];
            }
            if (src1_shape[j] != 1){
                src1_index = src1_index * src1_shape[j] + indices[j];
            }
            if (src2_shape[j] != 1){
                src2_index = src2_index * src2_shape[j] + indices[j];
            }
        }
        dst_[i] = condition_[condition_index] ? src1_[src1_index] : src2_[src2_index];
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
    void *dst, 
    void *src1,
    void *src2,
    void *condition,
    void *stream){
    if (desc->dtype == F16) {
        return where_cpu<uint16_t>(desc, dst, src1, src2, condition, stream);
    }
    if (desc->dtype == F32) {
        return where_cpu<float>(desc, dst, src1, src2, condition, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}