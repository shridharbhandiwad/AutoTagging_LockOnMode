#include "gating.h"
#include <cmath>
#include <limits>

double Gating::mahalanobisDistance(
    const Position& measurement,
    const Position& prediction,
    const std::array<std::array<double, 3>, 3>& covariance) {
    
    // Innovation vector
    std::array<double, 3> diff = {
        measurement[0] - prediction[0],
        measurement[1] - prediction[1],
        measurement[2] - prediction[2]
    };
    
    // Invert covariance matrix (simplified 3x3)
    double det = covariance[0][0] * (covariance[1][1]*covariance[2][2] - covariance[1][2]*covariance[2][1]) -
                 covariance[0][1] * (covariance[1][0]*covariance[2][2] - covariance[1][2]*covariance[2][0]) +
                 covariance[0][2] * (covariance[1][0]*covariance[2][1] - covariance[1][1]*covariance[2][0]);
    
    if (std::abs(det) < 1e-10) {
        // Singular matrix, return large distance
        return 1e6;
    }
    
    std::array<std::array<double, 3>, 3> inv;
    inv[0][0] = (covariance[1][1]*covariance[2][2] - covariance[1][2]*covariance[2][1]) / det;
    inv[0][1] = (covariance[0][2]*covariance[2][1] - covariance[0][1]*covariance[2][2]) / det;
    inv[0][2] = (covariance[0][1]*covariance[1][2] - covariance[0][2]*covariance[1][1]) / det;
    inv[1][0] = (covariance[1][2]*covariance[2][0] - covariance[1][0]*covariance[2][2]) / det;
    inv[1][1] = (covariance[0][0]*covariance[2][2] - covariance[0][2]*covariance[2][0]) / det;
    inv[1][2] = (covariance[0][2]*covariance[1][0] - covariance[0][0]*covariance[1][2]) / det;
    inv[2][0] = (covariance[1][0]*covariance[2][1] - covariance[1][1]*covariance[2][0]) / det;
    inv[2][1] = (covariance[0][1]*covariance[2][0] - covariance[0][0]*covariance[2][1]) / det;
    inv[2][2] = (covariance[0][0]*covariance[1][1] - covariance[0][1]*covariance[1][0]) / det;
    
    // Compute d^2 = diff' * inv * diff
    double d_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        double temp = 0.0;
        for (int j = 0; j < 3; ++j) {
            temp += inv[i][j] * diff[j];
        }
        d_squared += diff[i] * temp;
    }
    
    return d_squared;
}

bool Gating::passesGate(
    const Position& measurement,
    const Position& prediction,
    const std::array<std::array<double, 3>, 3>& covariance,
    double gate_threshold) {
    
    double distance = mahalanobisDistance(measurement, prediction, covariance);
    return distance <= gate_threshold;
}

double Gating::computeAssociationCost(
    const Position& measurement,
    const Position& prediction,
    const std::array<std::array<double, 3>, 3>& covariance) {
    
    return mahalanobisDistance(measurement, prediction, covariance);
}

int Gating::findBestAssociation(
    const std::vector<Position>& measurements,
    const Position& prediction,
    const std::array<std::array<double, 3>, 3>& covariance,
    double gate_threshold) {
    
    int best_idx = -1;
    double best_cost = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < measurements.size(); ++i) {
        double cost = computeAssociationCost(measurements[i], prediction, covariance);
        
        if (cost <= gate_threshold && cost < best_cost) {
            best_cost = cost;
            best_idx = static_cast<int>(i);
        }
    }
    
    return best_idx;
}
