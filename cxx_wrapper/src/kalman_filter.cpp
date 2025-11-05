#include "kalman_filter.h"
#include <cmath>
#include <algorithm>

KalmanFilter::KalmanFilter() 
    : process_noise_(1.0), meas_noise_(1.0) {
    state_.fill(0.0);
    initializeCovariance();
}

KalmanFilter::KalmanFilter(const State& initial_state, double process_noise, double meas_noise)
    : state_(initial_state), process_noise_(process_noise), meas_noise_(meas_noise) {
    initializeCovariance();
}

void KalmanFilter::initializeCovariance() {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            covariance_[i][j] = (i == j) ? 10.0 : 0.0;
        }
    }
}

void KalmanFilter::predict(double dt) {
    // State transition: simple constant velocity model
    // x_k = F * x_{k-1}
    State predicted_state;
    predicted_state[0] = state_[0] + state_[3] * dt;  // x + vx*dt
    predicted_state[1] = state_[1] + state_[4] * dt;  // y + vy*dt
    predicted_state[2] = state_[2] + state_[5] * dt;  // z + vz*dt
    predicted_state[3] = state_[3];  // vx
    predicted_state[4] = state_[4];  // vy
    predicted_state[5] = state_[5];  // vz
    
    state_ = predicted_state;
    
    // Covariance prediction: P = F*P*F' + Q
    // Simplified: add process noise to diagonal
    for (int i = 0; i < STATE_DIM; ++i) {
        covariance_[i][i] += process_noise_ * dt;
    }
}

KalmanFilter::FilterResult KalmanFilter::update(const Measurement& measurement) {
    FilterResult result;
    
    // Measurement prediction (H * x)
    std::array<double, MEAS_DIM> predicted_meas = {
        state_[0],  // x
        state_[1],  // y
        state_[2]   // z
    };
    
    // Innovation (residual): y = z - H*x
    std::array<double, MEAS_DIM> innovation;
    for (int i = 0; i < MEAS_DIM; ++i) {
        innovation[i] = measurement[i] - predicted_meas[i];
    }
    
    // Innovation covariance: S = H*P*H' + R
    std::array<std::array<double, MEAS_DIM>, MEAS_DIM> S;
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            S[i][j] = covariance_[i][j] + ((i == j) ? meas_noise_ : 0.0);
        }
    }
    
    // Kalman gain: K = P*H' * S^{-1}
    // Simplified 3x3 matrix inversion
    double det = S[0][0] * (S[1][1]*S[2][2] - S[1][2]*S[2][1]) -
                 S[0][1] * (S[1][0]*S[2][2] - S[1][2]*S[2][0]) +
                 S[0][2] * (S[1][0]*S[2][1] - S[1][1]*S[2][0]);
    
    if (std::abs(det) < 1e-10) det = 1e-10;  // Avoid division by zero
    
    std::array<std::array<double, MEAS_DIM>, MEAS_DIM> S_inv;
    S_inv[0][0] = (S[1][1]*S[2][2] - S[1][2]*S[2][1]) / det;
    S_inv[0][1] = (S[0][2]*S[2][1] - S[0][1]*S[2][2]) / det;
    S_inv[0][2] = (S[0][1]*S[1][2] - S[0][2]*S[1][1]) / det;
    S_inv[1][0] = (S[1][2]*S[2][0] - S[1][0]*S[2][2]) / det;
    S_inv[1][1] = (S[0][0]*S[2][2] - S[0][2]*S[2][0]) / det;
    S_inv[1][2] = (S[0][2]*S[1][0] - S[0][0]*S[1][2]) / det;
    S_inv[2][0] = (S[1][0]*S[2][1] - S[1][1]*S[2][0]) / det;
    S_inv[2][1] = (S[0][1]*S[2][0] - S[0][0]*S[2][1]) / det;
    S_inv[2][2] = (S[0][0]*S[1][1] - S[0][1]*S[1][0]) / det;
    
    // Compute Kalman gain
    std::array<std::array<double, MEAS_DIM>, STATE_DIM> K;
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            K[i][j] = 0.0;
            for (int k = 0; k < MEAS_DIM; ++k) {
                K[i][j] += covariance_[i][k] * S_inv[k][j];
            }
        }
    }
    
    // State update: x = x + K*y
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            state_[i] += K[i][j] * innovation[j];
        }
    }
    
    // Covariance update: P = (I - K*H) * P
    Covariance I_KH;
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            I_KH[i][j] = (i == j) ? 1.0 : 0.0;
            if (j < MEAS_DIM) {
                I_KH[i][j] -= K[i][j];
            }
        }
    }
    
    Covariance new_cov;
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            new_cov[i][j] = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                new_cov[i][j] += I_KH[i][k] * covariance_[k][j];
            }
        }
    }
    covariance_ = new_cov;
    
    // Prepare result
    result.state = state_;
    result.covariance = covariance_;
    result.residual = innovation;
    result.innovation_magnitude = std::sqrt(
        innovation[0]*innovation[0] + 
        innovation[1]*innovation[1] + 
        innovation[2]*innovation[2]
    );
    
    return result;
}

void KalmanFilter::reset(const State& initial_state) {
    state_ = initial_state;
    initializeCovariance();
}
