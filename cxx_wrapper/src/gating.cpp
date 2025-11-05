/**
 * @file gating.cpp
 * @brief Gating implementation
 */

#include "gating.h"
#include <cmath>
#include <limits>

namespace tracking {

double Gating::mahalanobis_distance(
    const std::array<double, 3>& measurement,
    const std::vector<double>& predicted_state,
    const std::vector<double>& covariance) {
    
    if (predicted_state.size() < 3 || covariance.size() < 9) {
        return std::numeric_limits<double>::max();
    }
    
    // Innovation
    double dx = measurement[0] - predicted_state[0];
    double dy = measurement[1] - predicted_state[1];
    double dz = measurement[2] - predicted_state[2];
    
    // Inverse of 3x3 covariance matrix
    double det = covariance[0]*(covariance[4]*covariance[8] - covariance[5]*covariance[7]) -
                 covariance[1]*(covariance[3]*covariance[8] - covariance[5]*covariance[6]) +
                 covariance[2]*(covariance[3]*covariance[7] - covariance[4]*covariance[6]);
    
    if (std::abs(det) < 1e-10) {
        return std::numeric_limits<double>::max();
    }
    
    double inv[9];
    inv[0] = (covariance[4]*covariance[8] - covariance[5]*covariance[7]) / det;
    inv[1] = -(covariance[1]*covariance[8] - covariance[2]*covariance[7]) / det;
    inv[2] = (covariance[1]*covariance[5] - covariance[2]*covariance[4]) / det;
    inv[3] = -(covariance[3]*covariance[8] - covariance[5]*covariance[6]) / det;
    inv[4] = (covariance[0]*covariance[8] - covariance[2]*covariance[6]) / det;
    inv[5] = -(covariance[0]*covariance[5] - covariance[2]*covariance[3]) / det;
    inv[6] = (covariance[3]*covariance[7] - covariance[4]*covariance[6]) / det;
    inv[7] = -(covariance[0]*covariance[7] - covariance[1]*covariance[6]) / det;
    inv[8] = (covariance[0]*covariance[4] - covariance[1]*covariance[3]) / det;
    
    // d^2 = [dx, dy, dz] * inv * [dx, dy, dz]^T
    double temp[3];
    temp[0] = inv[0]*dx + inv[1]*dy + inv[2]*dz;
    temp[1] = inv[3]*dx + inv[4]*dy + inv[5]*dz;
    temp[2] = inv[6]*dx + inv[7]*dy + inv[8]*dz;
    
    double dist_sq = dx*temp[0] + dy*temp[1] + dz*temp[2];
    
    return std::sqrt(std::max(0.0, dist_sq));
}

bool Gating::is_within_gate(double distance, double gate_threshold) {
    return distance <= gate_threshold;
}

std::vector<double> Gating::compute_cost_matrix(
    const std::vector<double>& measurements,
    const std::vector<double>& tracks,
    int num_measurements,
    int num_tracks) {
    
    std::vector<double> cost_matrix(num_measurements * num_tracks);
    
    for (int i = 0; i < num_measurements; ++i) {
        for (int j = 0; j < num_tracks; ++j) {
            // Simple Euclidean distance for cost
            // measurements: [x1, y1, z1, x2, y2, z2, ...]
            // tracks: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
            
            double mx = measurements[i * 3 + 0];
            double my = measurements[i * 3 + 1];
            double mz = measurements[i * 3 + 2];
            
            double tx = tracks[j * 6 + 0];
            double ty = tracks[j * 6 + 1];
            double tz = tracks[j * 6 + 2];
            
            double dx = mx - tx;
            double dy = my - ty;
            double dz = mz - tz;
            
            cost_matrix[i * num_tracks + j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
    
    return cost_matrix;
}

} // namespace tracking
