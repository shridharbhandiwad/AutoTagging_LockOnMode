/**
 * @file bindings.cpp
 * @brief Python bindings using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kalman_filter.h"
#include "gating.h"

namespace py = pybind11;
using namespace tracking;

PYBIND11_MODULE(cxxlib, m) {
    m.doc() = "C++ tracking algorithms for airborne track behavior tagging";
    
    // KalmanFilter class
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>())
        .def("initialize", &KalmanFilter::initialize,
             py::arg("position"), py::arg("velocity"),
             "Initialize filter with initial state")
        .def("predict", &KalmanFilter::predict,
             py::arg("dt"),
             "Predict next state")
        .def("update", &KalmanFilter::update,
             py::arg("measurement"), py::arg("measurement_noise"),
             "Update with measurement")
        .def("get_state", &KalmanFilter::get_state,
             "Get current state estimate")
        .def("get_covariance", &KalmanFilter::get_covariance,
             "Get state covariance matrix")
        .def("get_innovation", &KalmanFilter::get_innovation,
             "Get innovation from last update")
        .def("reset", &KalmanFilter::reset,
             "Reset filter");
    
    // Gating class (static methods)
    py::class_<Gating>(m, "Gating")
        .def_static("mahalanobis_distance", &Gating::mahalanobis_distance,
                   py::arg("measurement"), py::arg("predicted_state"), py::arg("covariance"),
                   "Compute Mahalanobis distance")
        .def_static("is_within_gate", &Gating::is_within_gate,
                   py::arg("distance"), py::arg("gate_threshold") = 9.21,
                   "Check if measurement is within gate")
        .def_static("compute_cost_matrix", &Gating::compute_cost_matrix,
                   py::arg("measurements"), py::arg("tracks"),
                   py::arg("num_measurements"), py::arg("num_tracks"),
                   "Compute association cost matrix");
    
    // Convenience function for batch Kalman filtering
    m.def("run_kalman", [](py::array_t<double> measurements, double dt) {
        auto buf = measurements.request();
        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("Measurements must be Nx3 array");
        }
        
        int n = buf.shape[0];
        double* ptr = static_cast<double*>(buf.ptr);
        
        KalmanFilter kf;
        
        // Initialize with first measurement
        std::array<double, 3> init_pos = {ptr[0], ptr[1], ptr[2]};
        std::array<double, 3> init_vel = {0.0, 0.0, 0.0};
        kf.initialize(init_pos, init_vel);
        
        std::vector<std::vector<double>> states;
        std::vector<std::vector<double>> innovations;
        
        for (int i = 0; i < n; ++i) {
            kf.predict(dt);
            
            std::array<double, 3> meas = {
                ptr[i*3 + 0],
                ptr[i*3 + 1],
                ptr[i*3 + 2]
            };
            std::array<double, 3> noise = {1.0, 1.0, 1.0};
            
            kf.update(meas, noise);
            
            states.push_back(kf.get_state());
            innovations.push_back(kf.get_innovation());
        }
        
        py::dict result;
        result["states"] = states;
        result["innovations"] = innovations;
        result["final_covariance"] = kf.get_covariance();
        
        return result;
    }, py::arg("measurements"), py::arg("dt") = 0.1,
       "Run Kalman filter on measurement sequence");
}
