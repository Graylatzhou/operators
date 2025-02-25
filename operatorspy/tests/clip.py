from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_bool
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch
from typing import Tuple
import numpy as np

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(input, min, max):
    return torch.clamp(input, min, max)


def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    min,
    max,
    tensor_dtype=torch.float16
):
    print(
        f"Testing clip on {torch_device} with x_shape:{x_shape} dtype:{tensor_dtype} max:{max} min:{min}"
    )
    x = torch.randn(x_shape, dtype=tensor_dtype, device=torch_device)
    output = torch.randn(x_shape, dtype=tensor_dtype, device=torch_device)
    if min != None:
        min = torch.tensor(min, dtype=torch.float32, device=torch_device)
    else:
        min = torch.tensor(float("-inf"), dtype=torch.float32, device=torch_device)
    if max != None:
        max = torch.tensor(max, dtype=torch.float32, device=torch_device)
    else:
        max = torch.tensor(float("inf"), dtype=torch.float32, device=torch_device)
    for i in range(NUM_PRERUN if PROFILE else 1):
        if min == None and max == None:
            break
        ans = clip(x, min, max)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = clip(x, min, max)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :10f}")
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(output, lib)
    descriptor = infiniopClipDescriptor_t()
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            x_tensor.descriptor,
            y_tensor.descriptor,
        )
    )
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopClip(
                descriptor,
                x_tensor.data,
                min.data_ptr() if min != None else None,
                max.data_ptr() if max != None else None,
                y_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
                check_error(
                    lib.infiniopClip(
                    descriptor,
                    x_tensor.data,
                    min.data_ptr() if min != None else None,
                    max.data_ptr() if max != None else None,
                    y_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"lib time: {elapsed :10f}")
    print("x:", x)
    print("custom op ans:", output)
    print("ans:", ans) if max != None or min != None else print("ans:", x)
    assert torch.allclose(output, ans, atol=0, rtol=0) if max != None or min != None else torch.allclose(output, x, atol=0, rtol=0)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, min, max in test_cases:
        test(lib, handle, "cpu", x_shape, min, max, tensor_dtype=torch.float16)
        print("\n")
        #test(lib, handle, "cpu", x_shape, axes, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        ((3, 4), -1, 1),
        ((3, 4), None, 1),
        ((3, 4), -1, None),
        ((3, 4), None, None)
        # stride = 
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [infiniopClipDescriptor_t]
    test_cpu(lib, test_cases)
    print("All tests passed!")