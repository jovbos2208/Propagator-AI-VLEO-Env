#ifndef GRAVITY_H
#define GRAVITY_H

#include <Eigen/Dense>

namespace vleo_aerodynamics_core {

struct GravityParams {
    double mu = 3.986004418e14; // [m^3/s^2]
    double Re = 6378137.0;      // [m]
    double J2 = 1.08262668e-3;
    double J3 = -2.532153e-6;   // placeholder typical value
    double J4 = -1.6196216e-6;  // placeholder typical value
};

// Acceleration in ECI including central term and zonal harmonics J2–J4 (axis aligned with ECI z).
Eigen::Vector3d gravity_accel_eci(const Eigen::Vector3d& r_eci, const GravityParams& gp);

} // namespace vleo_aerodynamics_core

#endif // GRAVITY_H
