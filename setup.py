"""CUDA extension build. All metadata in pyproject.toml."""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import pybind11

cuda_dir = "splade_classifier/_cuda"

ext_modules = [
    CUDAExtension(
        name="splade_classifier._cuda.splade_cuda_kernels",
        sources=[f"{cuda_dir}/bindings.cpp", f"{cuda_dir}/splade_kernels.cu"],
        include_dirs=[cuda_dir, pybind11.get_include()],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--use_fast_math", "-Xcompiler", "-fno-strict-aliasing"],
        },
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension})
