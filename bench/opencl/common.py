# bench/opencl/common.py
# Common OpenCL utility functions

import os, time
import pyopencl as cl

def pickOpenCLDevice():
    platforms = cl.get_platforms()
    print("Found OpenCL platforms:")
    for i,p in enumerate(platforms):
        print(f"  [{i}] {p.name} - {p.vendor}")
        for j,dev in enumerate(p.get_devices()):
            print(f"     ({j}) {dev.name} type={cl.device_type.to_string(dev.type)}")

        for p in platforms:
            for dev in p.get_devices():
                if dev.type & cl.device_type.GPU:
                    print(f"Picking platform {p.name}, device {dev.name}")
                    return p, dev
    raise RuntimeError("No OpenCL GPU device found!")

def buildOpenCLProgramFromPath(ctx, src_path):
    with open(src_path, 'r') as f:
        src = f.read()
    program = cl.Program(ctx, src).build()
    return program

def buildOpenClProgramFromString(ctx, src):
    program = cl.Program(ctx, src).build()
    return program