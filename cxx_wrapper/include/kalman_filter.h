#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <vector>
#include <array>

/**
 * Simple 6-state Kalman filter for tracking
 * State: [x, y, z, vx, vy, vz]
 */
class KalmanFilter {
public:
    static constexpr int STATE_DIM = 6;
    static constexpr int MEAS_DIM = 3;
    
    using State = std::array<double, STATE_DIM>;
    using Measurement = std::array<double, MEAS_DIM>;
    using Covariance = std::array<std::array<double, STATE_DIM>, STATE_DIM>;
    
    struct FilterResult {
        State state;
        Covariance covariance;
        std::array<double, MEAS_DIM> residual;
        double innovation_magnitude;
    };
    
    KalmanFilter();
    KalmanFilter(const State& initial_state, double process_noise, double meas_noise);
    
    // Predict step
    void predict(double dt);
    
    // Update step with measurement
    FilterResult update(const Measurement& measurement);
    
    // Get current state
    State getState() const { return state_; }
    Covariance getCovariance() const { return covariance_; }
    
    // Reset filter
    void reset(const State& initial_state);
    
private:
    State state_;
    Covariance covariance_;
    double process_noise_;
    double meas_noise_;
    
    void initializeCovariance();
};

#endif // KALMAN_FILTER_H
