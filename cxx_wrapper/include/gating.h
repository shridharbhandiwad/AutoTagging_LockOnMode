#ifndef GATING_H
#define GATING_H

#include <array>
#include <vector>

/**
 * Gating and association functions for tracking
 */
class Gating {
public:
    using Position = std::array<double, 3>;
    
    /**
     * Compute Mahalanobis distance between measurement and predicted position
     */
    static double mahalanobisDistance(
        const Position& measurement,
        const Position& prediction,
        const std::array<std::array<double, 3>, 3>& covariance
    );
    
    /**
     * Check if measurement passes gate (within threshold)
     */
    static bool passesGate(
        const Position& measurement,
        const Position& prediction,
        const std::array<std::array<double, 3>, 3>& covariance,
        double gate_threshold = 9.21  // Chi-square 99% for 3 DOF
    );
    
    /**
     * Compute association cost between track and measurement
     */
    static double computeAssociationCost(
        const Position& measurement,
        const Position& prediction,
        const std::array<std::array<double, 3>, 3>& covariance
    );
    
    /**
     * Find best measurement association for a track
     * Returns index of best measurement, or -1 if none pass gate
     */
    static int findBestAssociation(
        const std::vector<Position>& measurements,
        const Position& prediction,
        const std::array<std::array<double, 3>, 3>& covariance,
        double gate_threshold = 9.21
    );
};

#endif // GATING_H
