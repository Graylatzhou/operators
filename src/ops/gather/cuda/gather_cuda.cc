#include "gather_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                            GatherCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t indices,
                                            int64_t axis
                                            ){
    if (y->dt != x->dt){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != F16 && y->dt != F32){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)){
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (axis < 0 || axis >= x->ndim){
        return STATUS_BAD_PARAM;
    }
}

infiniopStatus_t gather_nv_gpu(
    GatherCudaDescriptor_t desc,
    void *x,
    void *indices,
    void *y,
    void *stream){
    return STATUS_SUCCESS;
}


infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                            void *x,
                            void *indices,
                            void *y,
                            void *stream){
    if (desc->dtype == F16){

    }
    if (desc->dtype == F32){

    }
    return STATUS_BAD_DEVICE;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc){
    return STATUS_SUCCESS;
}