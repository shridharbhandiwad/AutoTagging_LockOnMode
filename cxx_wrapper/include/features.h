#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <array>

/**
 * Feature extraction utilities for track analysis
 */
class FeatureExtractor {
public:
    using TimeSeries = std::vector<double>;
    using Position3D = std::array<double, 3>;
    
    struct TrackFeatures {
        // Speed statistics
        double max_speed;
        double min_speed;
        double mean_speed;
        double std_speed;
        
        // Height statistics
        double max_height;
        double min_height;
        double mean_height;
        
        // Range statistics
        double max_range;
        double min_range;
        double mean_range;
        
        // Maneuver indicators
        double maneuver_index;  // Based on acceleration changes
        double curvature;
        double jerk_magnitude;
        
        // Signal quality
        double snr_mean;
        double rcs_mean;
        
        // Temporal
        double flight_time;
    };
    
    /**
     * Extract all features from track time series
     */
    static TrackFeatures extractFeatures(
        const std::vector<Position3D>& positions,
        const std::vector<Position3D>& velocities,
        const std::vector<Position3D>& accelerations,
        const TimeSeries& snr_values,
        const TimeSeries& rcs_values,
        const TimeSeries& timestamps
    );
    
    /**
     * Compute speed from velocity vector
     */
    static double computeSpeed(const Position3D& velocity);
    
    /**
     * Compute range from position
     */
    static double computeRange(const Position3D& position);
    
    /**
     * Compute maneuver index from acceleration time series
     */
    static double computeManeuverIndex(const std::vector<Position3D>& accelerations);
    
    /**
     * Compute path curvature
     */
    static double computeCurvature(const std::vector<Position3D>& positions);
    
    /**
     * Compute jerk (derivative of acceleration)
     */
    static double computeJerk(
        const std::vector<Position3D>& accelerations,
        const TimeSeries& timestamps
    );
};

#endif // FEATURES_H
