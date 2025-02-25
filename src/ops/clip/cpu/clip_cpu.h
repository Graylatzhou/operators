#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t element_num;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                        ClipCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t x,
                                        infiniopTensorDescriptor_t y
                                        );

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                                  void const *x, 
                                  void *min,
                                  void *max,
                                  void *y,
                                  void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
