import shutil
import subprocess
import os
import sys
from shutil import copytree, ignore_patterns
from functools import reduce

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))

includes = [
    "aten/*",
    "torch/*"
]

# List of operators currently disabled
yaml_file = os.path.join(amd_build_dir, "disabled_features.yaml")

# Apply patch files in place.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=proj_dir)

# HIPCC Compiler doesn't provide host defines - Automatically include them.
for root, _, files in os.walk(os.path.join(proj_dir, "aten/src/ATen")):
    for filename in files:
        if filename.endswith(".cu") or filename.endswith(".cuh"):
            filepath = os.path.join(root, filename)

            # Add the include header!
            with open(filepath, "r+") as f:
                txt = f.read()
                result = '#include "hip/hip_runtime.h"\n%s' % txt
                f.seek(0)
                f.write(result)
                f.truncate()
                f.flush()

                # Flush to disk
                os.fsync(f)

# Make various replacements inside AMD_BUILD/torch directory
ignore_files = ["csrc/autograd/profiler.h", "csrc/autograd/profiler.cpp",
                "csrc/cuda/cuda_check.h", "csrc/jit/fusion_compiler.cpp"]
for root, _directories, files in os.walk(os.path.join(proj_dir, "torch")):
    for filename in files:
        if filename.endswith(".cpp") or filename.endswith(".h"):
            source = os.path.join(root, filename)
            # Disabled files
            if reduce(lambda result, exclude: source.endswith(exclude) or result, ignore_files, False):
                continue
            # Update contents.
            with open(source, "r+") as f:
                contents = f.read()
                contents = contents.replace("USE_CUDA", "USE_ROCM")
                contents = contents.replace("CUDA_VERSION", "0")
                f.seek(0)
                f.write(contents)
                f.truncate()
                f.flush()
                os.fsync(f)

# Execute the Hipify Script.
args = (["--project-directory", proj_dir] +
        ["--output-directory", proj_dir] +
        ["--includes"] + includes +
        ["--yaml-settings", yaml_file] +
        ["--add-static-casts", "True"] +
        ["--show-progress", "False"])

subprocess.check_call([
    sys.executable,
    os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py")
] + args)
