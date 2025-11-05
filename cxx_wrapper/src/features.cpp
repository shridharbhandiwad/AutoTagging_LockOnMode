#include "features.h"
#include <cmath>
#include <algorithm>
#include <numeric>

double FeatureExtractor::computeSpeed(const Position3D& velocity) {
    return std::sqrt(
        velocity[0]*velocity[0] + 
        velocity[1]*velocity[1] + 
        velocity[2]*velocity[2]
    );
}

double FeatureExtractor::computeRange(const Position3D& position) {
    return std::sqrt(
        position[0]*position[0] + 
        position[1]*position[1] + 
        position[2]*position[2]
    );
}

double FeatureExtractor::computeManeuverIndex(const std::vector<Position3D>& accelerations) {
    if (accelerations.size() < 2) return 0.0;
    
    double total_change = 0.0;
    for (size_t i = 1; i < accelerations.size(); ++i) {
        double change = std::sqrt(
            std::pow(accelerations[i][0] - accelerations[i-1][0], 2) +
            std::pow(accelerations[i][1] - accelerations[i-1][1], 2) +
            std::pow(accelerations[i][2] - accelerations[i-1][2], 2)
        );
        total_change += change;
    }
    
    return total_change / accelerations.size();
}

double FeatureExtractor::computeCurvature(const std::vector<Position3D>& positions) {
    if (positions.size() < 3) return 0.0;
    
    double total_curvature = 0.0;
    int count = 0;
    
    for (size_t i = 1; i < positions.size() - 1; ++i) {
        // Vectors from point i-1 to i and i to i+1
        double v1_x = positions[i][0] - positions[i-1][0];
        double v1_y = positions[i][1] - positions[i-1][1];
        double v1_z = positions[i][2] - positions[i-1][2];
        
        double v2_x = positions[i+1][0] - positions[i][0];
        double v2_y = positions[i+1][1] - positions[i][1];
        double v2_z = positions[i+1][2] - positions[i][2];
        
        // Cross product magnitude
        double cross_x = v1_y * v2_z - v1_z * v2_y;
        double cross_y = v1_z * v2_x - v1_x * v2_z;
        double cross_z = v1_x * v2_y - v1_y * v2_x;
        double cross_mag = std::sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);
        
        // Magnitudes of v1 and v2
        double v1_mag = std::sqrt(v1_x*v1_x + v1_y*v1_y + v1_z*v1_z);
        double v2_mag = std::sqrt(v2_x*v2_x + v2_y*v2_y + v2_z*v2_z);
        
        if (v1_mag > 1e-6 && v2_mag > 1e-6) {
            double curvature = cross_mag / (v1_mag * v2_mag);
            total_curvature += curvature;
            count++;
        }
    }
    
    return count > 0 ? total_curvature / count : 0.0;
}

double FeatureExtractor::computeJerk(
    const std::vector<Position3D>& accelerations,
    const TimeSeries& timestamps) {
    
    if (accelerations.size() < 2 || timestamps.size() < 2) return 0.0;
    
    double total_jerk = 0.0;
    int count = 0;
    
    for (size_t i = 1; i < accelerations.size(); ++i) {
        double dt = timestamps[i] - timestamps[i-1];
        if (dt > 1e-6) {
            double jerk_x = (accelerations[i][0] - accelerations[i-1][0]) / dt;
            double jerk_y = (accelerations[i][1] - accelerations[i-1][1]) / dt;
            double jerk_z = (accelerations[i][2] - accelerations[i-1][2]) / dt;
            
            double jerk_mag = std::sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z);
            total_jerk += jerk_mag;
            count++;
        }
    }
    
    return count > 0 ? total_jerk / count : 0.0;
}

FeatureExtractor::TrackFeatures FeatureExtractor::extractFeatures(
    const std::vector<Position3D>& positions,
    const std::vector<Position3D>& velocities,
    const std::vector<Position3D>& accelerations,
    const TimeSeries& snr_values,
    const TimeSeries& rcs_values,
    const TimeSeries& timestamps) {
    
    TrackFeatures features = {};
    
    if (positions.empty() || velocities.empty()) {
        return features;
    }
    
    // Speed statistics
    std::vector<double> speeds;
    for (const auto& vel : velocities) {
        speeds.push_back(computeSpeed(vel));
    }
    
    if (!speeds.empty()) {
        features.max_speed = *std::max_element(speeds.begin(), speeds.end());
        features.min_speed = *std::min_element(speeds.begin(), speeds.end());
        features.mean_speed = std::accumulate(speeds.begin(), speeds.end(), 0.0) / speeds.size();
        
        // Standard deviation
        double var = 0.0;
        for (double s : speeds) {
            var += (s - features.mean_speed) * (s - features.mean_speed);
        }
        features.std_speed = std::sqrt(var / speeds.size());
    }
    
    // Height statistics (z-coordinate)
    std::vector<double> heights;
    for (const auto& pos : positions) {
        heights.push_back(pos[2]);
    }
    
    if (!heights.empty()) {
        features.max_height = *std::max_element(heights.begin(), heights.end());
        features.min_height = *std::min_element(heights.begin(), heights.end());
        features.mean_height = std::accumulate(heights.begin(), heights.end(), 0.0) / heights.size();
    }
    
    // Range statistics
    std::vector<double> ranges;
    for (const auto& pos : positions) {
        ranges.push_back(computeRange(pos));
    }
    
    if (!ranges.empty()) {
        features.max_range = *std::max_element(ranges.begin(), ranges.end());
        features.min_range = *std::min_element(ranges.begin(), ranges.end());
        features.mean_range = std::accumulate(ranges.begin(), ranges.end(), 0.0) / ranges.size();
    }
    
    // Maneuver indicators
    if (!accelerations.empty()) {
        features.maneuver_index = computeManeuverIndex(accelerations);
    }
    
    if (positions.size() >= 3) {
        features.curvature = computeCurvature(positions);
    }
    
    if (!accelerations.empty() && !timestamps.empty()) {
        features.jerk_magnitude = computeJerk(accelerations, timestamps);
    }
    
    // Signal quality
    if (!snr_values.empty()) {
        features.snr_mean = std::accumulate(snr_values.begin(), snr_values.end(), 0.0) / snr_values.size();
    }
    
    if (!rcs_values.empty()) {
        features.rcs_mean = std::accumulate(rcs_values.begin(), rcs_values.end(), 0.0) / rcs_values.size();
    }
    
    // Flight time
    if (timestamps.size() >= 2) {
        features.flight_time = timestamps.back() - timestamps.front();
    }
    
    return features;
}
