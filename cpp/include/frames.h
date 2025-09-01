#ifndef FRAMES_H
#define FRAMES_H

#include <Eigen/Dense>

namespace vleo_aerodynamics_core {

// WGS-84 constants
constexpr double WGS84_A = 6378137.0;           // semi-major axis [m]
constexpr double WGS84_F = 1.0 / 298.257223563; // flattening
constexpr double WGS84_B = WGS84_A * (1.0 - WGS84_F);
constexpr double WGS84_E2 = (WGS84_A*WGS84_A - WGS84_B*WGS84_B) / (WGS84_A*WGS84_A);

// Earth rotation rate [rad/s]
constexpr double OMEGA_EARTH = 7.2921150e-5;

// Julian date utilities
double jd_from_unix(double unix_seconds);
double gmst_rad(double jd_ut1);

// ECI <-> ECEF (ignoring polar motion and nutation for now)
Eigen::Matrix3d R_ECI_to_ECEF(double jd_ut1);
Eigen::Matrix3d R_ECEF_to_ECI(double jd_ut1);

// ECEF to geodetic (lat rad, lon rad, alt m)
void ecef_to_geodetic(const Eigen::Vector3d& r_ecef,
                      double& lat_rad, double& lon_rad, double& alt_m);

} // namespace vleo_aerodynamics_core

#endif // FRAMES_H

