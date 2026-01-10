import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define compilation flags
# -O3: Max optimization
# --use_fast_math: Uses hardware intrinsics for math (faster, slightly less precise)
# -gencode=arch=compute_75: Targets NVIDIA Tesla T4 (Turing Architecture)
cxx_args = ['-O3']
nvcc_args = [
    '-O3', 
    '--use_fast_math', 
    '-gencode=arch=compute_75,code=sm_75'
]

setup(
    name='smol_kernels',
    version='0.1.0',
    author='Umang Singh',
    author_email='umang2001umang@gmail.com',
    description='High-performance CUDA inference kernels for Llama/SmolLM on Tesla T4',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/Umang-projects/SmolInference-T4',
    ext_modules=[
        CUDAExtension(
            name='smol_kernels',
            sources=[
                'csrc/bindings.cpp',
                'csrc/kernels.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'transformers'
    ],
)