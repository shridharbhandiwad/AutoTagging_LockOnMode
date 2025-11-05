/**
 * @file kalman_filter.cpp
 * @brief Kalman filter implementation
 */

#include "kalman_filter.h"
#include <cmath>
#include <algorithm>

namespace tracking {

KalmanFilter::KalmanFilter()
    : process_noise_(0.1), initialized_(false) {
    reset();
}

void KalmanFilter::initialize(const std::array<double, 3>& position,
                              const std::array<double, 3>& velocity) {
    state_[0] = position[0];
    state_[1] = position[1];
    state_[2] = position[2];
    state_[3] = velocity[0];
    state_[4] = velocity[1];
    state_[5] = velocity[2];
    
    // Initialize covariance to identity * 10
    std::fill(covariance_.begin(), covariance_.end(), 0.0);
    for (int i = 0; i < STATE_DIM; ++i) {
        covariance_[i * STATE_DIM + i] = 10.0;
    }
    
    initialized_ = true;
}

void KalmanFilter::predict(double dt) {
    if (!initialized_) return;
    
    // State transition: x_k = F * x_{k-1}
    // F = [I_3, dt*I_3]
    //     [0,   I_3   ]
    
    std::array<double, STATE_DIM> new_state = state_;
    
    // Update position: x = x + vx*dt
    new_state[0] = state_[0] + state_[3] * dt;
    new_state[1] = state_[1] + state_[4] * dt;
    new_state[2] = state_[2] + state_[5] * dt;
    
    state_ = new_state;
    
    // Update covariance: P_k = F * P_{k-1} * F^T + Q
    // Simplified: add process noise to diagonal
    for (int i = 0; i < STATE_DIM; ++i) {
        covariance_[i * STATE_DIM + i] += process_noise_ * dt;
    }
}

void KalmanFilter::update(const std::array<double, 3>& measurement,
                         const std::array<double, 3>& measurement_noise) {
    if (!initialized_) return;
    
    // Innovation: y = z - H*x
    // H = [I_3, 0]
    innovation_[0] = measurement[0] - state_[0];
    innovation_[1] = measurement[1] - state_[1];
    innovation_[2] = measurement[2] - state_[2];
    
    // Innovation covariance: S = H*P*H^T + R
    std::array<double, MEAS_DIM * MEAS_DIM> S;
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            S[i * MEAS_DIM + j] = covariance_[i * STATE_DIM + j];
            if (i == j) {
                S[i * MEAS_DIM + j] += measurement_noise[i];
            }
        }
    }
    
    // Kalman gain: K = P*H^T*S^{-1}
    // Simplified 3x3 inverse for S
    double det = S[0]*(S[4]*S[8] - S[5]*S[7]) -
                 S[1]*(S[3]*S[8] - S[5]*S[6]) +
                 S[2]*(S[3]*S[7] - S[4]*S[6]);
    
    if (std::abs(det) < 1e-10) return;  // Singular matrix
    
    std::array<double, MEAS_DIM * MEAS_DIM> S_inv;
    S_inv[0] = (S[4]*S[8] - S[5]*S[7]) / det;
    S_inv[1] = -(S[1]*S[8] - S[2]*S[7]) / det;
    S_inv[2] = (S[1]*S[5] - S[2]*S[4]) / det;
    S_inv[3] = -(S[3]*S[8] - S[5]*S[6]) / det;
    S_inv[4] = (S[0]*S[8] - S[2]*S[6]) / det;
    S_inv[5] = -(S[0]*S[5] - S[2]*S[3]) / det;
    S_inv[6] = (S[3]*S[7] - S[4]*S[6]) / det;
    S_inv[7] = -(S[0]*S[7] - S[1]*S[6]) / det;
    S_inv[8] = (S[0]*S[4] - S[1]*S[3]) / det;
    
    // K = P*H^T*S_inv (H^T just selects first 3 columns of P)
    std::array<double, STATE_DIM * MEAS_DIM> K;
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            K[i * MEAS_DIM + j] = 0.0;
            for (int k = 0; k < MEAS_DIM; ++k) {
                K[i * MEAS_DIM + j] += covariance_[i * STATE_DIM + k] * S_inv[k * MEAS_DIM + j];
            }
        }
    }
    
    // Update state: x = x + K*y
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            state_[i] += K[i * MEAS_DIM + j] * innovation_[j];
        }
    }
    
    // Update covariance: P = (I - K*H)*P
    // Simplified: just reduce diagonal
    for (int i = 0; i < MEAS_DIM; ++i) {
        covariance_[i * STATE_DIM + i] *= 0.9;
    }
}

std::vector<double> KalmanFilter::get_state() const {
    return std::vector<double>(state_.begin(), state_.end());
}

std::vector<double> KalmanFilter::get_covariance() const {
    return std::vector<double>(covariance_.begin(), covariance_.end());
}

std::vector<double> KalmanFilter::get_innovation() const {
    return std::vector<double>(innovation_.begin(), innovation_.end());
}

void KalmanFilter::reset() {
    std::fill(state_.begin(), state_.end(), 0.0);
    std::fill(covariance_.begin(), covariance_.end(), 0.0);
    std::fill(innovation_.begin(), innovation_.end(), 0.0);
    initialized_ = false;
}

} // namespace tracking
