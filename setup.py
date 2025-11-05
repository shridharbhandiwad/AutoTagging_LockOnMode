from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

ext_modules = [
    Pybind11Extension(
        "cxxlib",
        ["cxx_wrapper/src/bindings.cpp",
         "cxx_wrapper/src/kalman_filter.cpp",
         "cxx_wrapper/src/gating.cpp"],
        include_dirs=["cxx_wrapper/include"],
        cxx_std=17,
    ),
]

setup(
    name="airborne-track-tagger",
    version="1.0.0",
    description="Automated Airborne Track Behavior Tagging Application",
    author="Airborne Tracker Team",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={
        "console_scripts": [
            "track-tagger-gui=gui.main:main",
            "track-simulator=simulator.main:main",
        ],
    },
    python_requires=">=3.8",
)
