from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Define C++ extension modules
ext_modules = [
    Pybind11Extension(
        "cxxlib",
        ["cxx_wrapper/src/bindings.cpp",
         "cxx_wrapper/src/kalman_filter.cpp",
         "cxx_wrapper/src/gating.cpp",
         "cxx_wrapper/src/features.cpp"],
        include_dirs=["cxx_wrapper/include"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="airborne-tracker-tagger",
    version="1.0.0",
    author="Airborne Tracker Team",
    description="Automated tagging of airborne track behavior for radar systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "construct>=2.10.0",
        "pyarrow>=12.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "PySide6>=6.5.0",
        "pyqtgraph>=0.13.0",
        "matplotlib>=3.7.0",
        "shap>=0.42.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
