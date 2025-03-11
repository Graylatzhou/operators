#ifndef __CUDA_GATHER_H__
#define __CUDA_GATHER_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <memory>

typedef struct GatherCudaDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t element_num;
} GatherCudaDescriptor;

typedef struct GatherCudaDescriptor *GatherCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                            GatherCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t indices,
                                            int64_t axis
                                            );


infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                            void *x,
                            void *indices,
                            void *y,
                            void *stream);

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc);

#endif// __CUDA_MATMUL_H__
