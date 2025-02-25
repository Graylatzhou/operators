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

PROFILE = True
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopGatherDescriptor_t = POINTER(GatherDescriptor)

def gather(input, indices, axis):
    np_input = input.numpy()
    np_indices = indices.numpy()
    np_output = np.take(np_input, np_indices, axis=axis)
    return torch.from_numpy(np_output)

def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def inferShape(input_shape, indices_shape, axis):
    output_shape = input_shape[:axis] + tuple(indices_shape) + input_shape[axis + 1:]
    return output_shape

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    indices_shape,
    axis,
    tensor_dtype=torch.float16
):
    print(
        f"Testing clip on {torch_device} with x_shape:{x_shape} dtype:{tensor_dtype}"
    )
    x = torch.randn(x_shape, dtype=tensor_dtype, device=torch_device)
    if len(x.shape) == 2:
        indices = torch.tensor(2, dtype=torch.int64, device=torch_device)
    elif len(x.shape) == 3:
        indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=torch_device)
    dst = torch.randn(inferShape(x_shape, indices.shape, axis), dtype=tensor_dtype, device=torch_device)
    ans = gather(x, indices, axis)
    axis = axis
    x_tensor = to_tensor(x, lib)
    indices_tensor = to_tensor(indices, lib)
    dst_tensor = to_tensor(dst, lib)
    descriptor = infiniopGatherDescriptor_t()
    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            dst_tensor.descriptor,
            x_tensor.descriptor,
            indices_tensor.descriptor,
            axis
        )
    )
    x_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()
    dst_tensor.descriptor.contents.invalidate()
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopGather(
                descriptor,
                x_tensor.data,
                indices_tensor.data,
                dst_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGather(
                    descriptor,
                    x_tensor.data,
                    indices_tensor.data,
                    dst_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"lib time: {elapsed :10f}")
    print(f"pytorch ans: {ans}")
    print(f"lib ans: {dst_tensor.data}")
    assert torch.allclose(dst, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, indices_shape, axis in test_cases:
        test(lib, handle, "cpu", x_shape, indices_shape, axis, tensor_dtype=torch.float16)
        print("\n")
        #test(lib, handle, "cpu", x_shape, axes, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        ((3, 4), (2), 0),
        ((2, 3, 4), (2, 2), 1),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [infiniopGatherDescriptor_t]
    test_cpu(lib, test_cases)
    print("All tests passed!")