#include "frames.h"
#include <cmath>

namespace vleo_aerodynamics_core {

double jd_from_unix(double unix_seconds) {
    // Unix epoch 1970-01-01T00:00:00Z corresponds to JD 2440587.5
    return 2440587.5 + unix_seconds / 86400.0;
}

double gmst_rad(double jd_ut1) {
    // IAU 1982 GMST approximation (Vallado)
    const double T = (jd_ut1 - 2451545.0) / 36525.0;
    double sec = 67310.54841
               + (876600.0 * 3600.0 + 8640184.812866) * T
               + 0.093104 * T * T
               - 6.2e-6 * T * T * T;
    sec = fmod(sec, 86400.0);
    if (sec < 0) sec += 86400.0;
    return sec * (2.0 * M_PI / 86400.0);
}

Eigen::Matrix3d Rz(double angle_rad) {
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    Eigen::Matrix3d R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    return R;
}

Eigen::Matrix3d R_ECI_to_ECEF(double jd_ut1) {
    return Rz(gmst_rad(jd_ut1));
}

Eigen::Matrix3d R_ECEF_to_ECI(double jd_ut1) {
    return Rz(-gmst_rad(jd_ut1));
}

void ecef_to_geodetic(const Eigen::Vector3d& r_ecef,
                      double& lat_rad, double& lon_rad, double& alt_m) {
    // Bowring's method
    double x = r_ecef.x();
    double y = r_ecef.y();
    double z = r_ecef.z();
    double lon = std::atan2(y, x);
    double p = std::sqrt(x*x + y*y);
    const double a = WGS84_A;
    const double e2 = WGS84_E2;
    const double b = WGS84_B;
    const double ep2 = (a*a - b*b) / (b*b);
    double th = std::atan2(a * z, b * p);
    double sin_th = std::sin(th);
    double cos_th = std::cos(th);
    double lat = std::atan2(z + ep2 * b * sin_th * sin_th * sin_th,
                            p - e2 * a * cos_th * cos_th * cos_th);
    double sin_lat = std::sin(lat);
    double N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    double alt = p / std::cos(lat) - N;
    lat_rad = lat;
    lon_rad = lon;
    alt_m = alt;
}

} // namespace vleo_aerodynamics_core

