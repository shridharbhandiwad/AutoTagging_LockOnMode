#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kalman_filter.h"
#include "gating.h"
#include "features.h"

namespace py = pybind11;

PYBIND11_MODULE(cxxlib, m) {
    m.doc() = "C++ algorithms for airborne track processing";
    
    // Kalman Filter bindings
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>())
        .def(py::init<const KalmanFilter::State&, double, double>(),
             py::arg("initial_state"),
             py::arg("process_noise") = 1.0,
             py::arg("meas_noise") = 1.0)
        .def("predict", &KalmanFilter::predict, py::arg("dt"))
        .def("update", &KalmanFilter::update, py::arg("measurement"))
        .def("get_state", &KalmanFilter::getState)
        .def("get_covariance", &KalmanFilter::getCovariance)
        .def("reset", &KalmanFilter::reset, py::arg("initial_state"));
    
    py::class_<KalmanFilter::FilterResult>(m, "FilterResult")
        .def_readonly("state", &KalmanFilter::FilterResult::state)
        .def_readonly("covariance", &KalmanFilter::FilterResult::covariance)
        .def_readonly("residual", &KalmanFilter::FilterResult::residual)
        .def_readonly("innovation_magnitude", &KalmanFilter::FilterResult::innovation_magnitude);
    
    // Gating bindings
    py::class_<Gating>(m, "Gating")
        .def_static("mahalanobis_distance", &Gating::mahalanobisDistance,
                   py::arg("measurement"),
                   py::arg("prediction"),
                   py::arg("covariance"))
        .def_static("passes_gate", &Gating::passesGate,
                   py::arg("measurement"),
                   py::arg("prediction"),
                   py::arg("covariance"),
                   py::arg("gate_threshold") = 9.21)
        .def_static("compute_association_cost", &Gating::computeAssociationCost,
                   py::arg("measurement"),
                   py::arg("prediction"),
                   py::arg("covariance"))
        .def_static("find_best_association", &Gating::findBestAssociation,
                   py::arg("measurements"),
                   py::arg("prediction"),
                   py::arg("covariance"),
                   py::arg("gate_threshold") = 9.21);
    
    // Feature Extractor bindings
    py::class_<FeatureExtractor::TrackFeatures>(m, "TrackFeatures")
        .def_readonly("max_speed", &FeatureExtractor::TrackFeatures::max_speed)
        .def_readonly("min_speed", &FeatureExtractor::TrackFeatures::min_speed)
        .def_readonly("mean_speed", &FeatureExtractor::TrackFeatures::mean_speed)
        .def_readonly("std_speed", &FeatureExtractor::TrackFeatures::std_speed)
        .def_readonly("max_height", &FeatureExtractor::TrackFeatures::max_height)
        .def_readonly("min_height", &FeatureExtractor::TrackFeatures::min_height)
        .def_readonly("mean_height", &FeatureExtractor::TrackFeatures::mean_height)
        .def_readonly("max_range", &FeatureExtractor::TrackFeatures::max_range)
        .def_readonly("min_range", &FeatureExtractor::TrackFeatures::min_range)
        .def_readonly("mean_range", &FeatureExtractor::TrackFeatures::mean_range)
        .def_readonly("maneuver_index", &FeatureExtractor::TrackFeatures::maneuver_index)
        .def_readonly("curvature", &FeatureExtractor::TrackFeatures::curvature)
        .def_readonly("jerk_magnitude", &FeatureExtractor::TrackFeatures::jerk_magnitude)
        .def_readonly("snr_mean", &FeatureExtractor::TrackFeatures::snr_mean)
        .def_readonly("rcs_mean", &FeatureExtractor::TrackFeatures::rcs_mean)
        .def_readonly("flight_time", &FeatureExtractor::TrackFeatures::flight_time);
    
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def_static("extract_features", &FeatureExtractor::extractFeatures,
                   py::arg("positions"),
                   py::arg("velocities"),
                   py::arg("accelerations"),
                   py::arg("snr_values"),
                   py::arg("rcs_values"),
                   py::arg("timestamps"))
        .def_static("compute_speed", &FeatureExtractor::computeSpeed,
                   py::arg("velocity"))
        .def_static("compute_range", &FeatureExtractor::computeRange,
                   py::arg("position"))
        .def_static("compute_maneuver_index", &FeatureExtractor::computeManeuverIndex,
                   py::arg("accelerations"))
        .def_static("compute_curvature", &FeatureExtractor::computeCurvature,
                   py::arg("positions"))
        .def_static("compute_jerk", &FeatureExtractor::computeJerk,
                   py::arg("accelerations"),
                   py::arg("timestamps"));
}
