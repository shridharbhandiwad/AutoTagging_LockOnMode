/**
 * @file kalman_filter.h
 * @brief Kalman filter implementation for track state estimation
 */

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <vector>
#include <array>

namespace tracking {

/**
 * @brief 3D Kalman filter for position and velocity tracking
 * 
 * State vector: [x, y, z, vx, vy, vz]
 */
class KalmanFilter {
public:
    KalmanFilter();
    ~KalmanFilter() = default;
    
    /**
     * @brief Initialize filter with initial state
     * @param position Initial position [x, y, z]
     * @param velocity Initial velocity [vx, vy, vz]
     */
    void initialize(const std::array<double, 3>& position,
                   const std::array<double, 3>& velocity);
    
    /**
     * @brief Predict next state
     * @param dt Time step in seconds
     */
    void predict(double dt);
    
    /**
     * @brief Update with measurement
     * @param measurement Measurement vector [x, y, z]
     * @param measurement_noise Measurement noise covariance
     */
    void update(const std::array<double, 3>& measurement,
               const std::array<double, 3>& measurement_noise);
    
    /**
     * @brief Get current state estimate
     * @return State vector [x, y, z, vx, vy, vz]
     */
    std::vector<double> get_state() const;
    
    /**
     * @brief Get state covariance matrix (flattened)
     * @return Covariance matrix as vector (row-major)
     */
    std::vector<double> get_covariance() const;
    
    /**
     * @brief Get innovation (residual) from last update
     * @return Innovation vector
     */
    std::vector<double> get_innovation() const;
    
    /**
     * @brief Reset filter
     */
    void reset();

private:
    static constexpr int STATE_DIM = 6;  // [x, y, z, vx, vy, vz]
    static constexpr int MEAS_DIM = 3;   // [x, y, z]
    
    std::array<double, STATE_DIM> state_;
    std::array<double, STATE_DIM * STATE_DIM> covariance_;
    std::array<double, MEAS_DIM> innovation_;
    
    double process_noise_;
    bool initialized_;
};

} // namespace tracking

#endif // KALMAN_FILTER_H
