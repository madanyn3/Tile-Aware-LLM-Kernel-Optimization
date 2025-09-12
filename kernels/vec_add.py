import pyopencl as cl
import numpy as np

n = 10
A = np.arange(n).astype(np.float32)
B = np.arange(n).astype(np.float32)
C = np.empty_like(A)

platforms = cl.get_platforms()
platform = platforms[0]                # pick first platform (AMD)
device = platform.get_devices()[0]     # pick first device (Radeon 860M)

print("Using platform:", platform.name)
print("Using device:", device.name)

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

with open("kernels/vec_add.cl", 'r') as f:
    kernel_code = f.read()

program = cl.Program(ctx, kernel_code).build()

mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

program.vec_add(queue, A.shape, None, A_buf, B_buf, C_buf, np.int32(n))
cl.enqueue_copy(queue, C, C_buf)

print("A:", A)
print("B:", B)
print("C:", C)