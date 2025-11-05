/**
 * @file gating.h
 * @brief Gating and data association utilities
 */

#ifndef GATING_H
#define GATING_H

#include <vector>
#include <array>

namespace tracking {

/**
 * @brief Gating functions for measurement-to-track association
 */
class Gating {
public:
    /**
     * @brief Compute Mahalanobis distance between measurement and predicted state
     * @param measurement Measurement vector [x, y, z]
     * @param predicted_state Predicted state [x, y, z, vx, vy, vz]
     * @param covariance Innovation covariance (flattened 3x3)
     * @return Mahalanobis distance
     */
    static double mahalanobis_distance(
        const std::array<double, 3>& measurement,
        const std::vector<double>& predicted_state,
        const std::vector<double>& covariance);
    
    /**
     * @brief Check if measurement is within gate
     * @param distance Mahalanobis distance
     * @param gate_threshold Gate threshold (e.g., chi-squared value)
     * @return True if within gate
     */
    static bool is_within_gate(double distance, double gate_threshold = 9.21);
    
    /**
     * @brief Compute association cost matrix
     * @param measurements List of measurements (flattened)
     * @param tracks List of track predictions (flattened)
     * @param num_measurements Number of measurements
     * @param num_tracks Number of tracks
     * @return Cost matrix (flattened row-major)
     */
    static std::vector<double> compute_cost_matrix(
        const std::vector<double>& measurements,
        const std::vector<double>& tracks,
        int num_measurements,
        int num_tracks);
};

} // namespace tracking

#endif // GATING_H
