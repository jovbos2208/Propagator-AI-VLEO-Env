#ifndef ATMO_NRLMSIS_H
#define ATMO_NRLMSIS_H

#include <Eigen/Dense>

// Placeholder interface for the NRLMSIS2.1 C-wrapper expected to be provided.
// The implementation should live in nrlmsis2.1_cpp and be linked in CMake when available.

struct AtmoOut {
    double rho_kg_m3 = 0.0;        // density
    double T_K = 0.0;              // temperature
    Eigen::Vector3d wind_ecef = Eigen::Vector3d::Zero(); // wind in ECEF frame (m/s)
    double mean_molecular_mass_kg = 16.0 * 1.6605390689252e-27; // optional; fallback if not provided
};

// Evaluate the atmosphere at UTC (Julian Date), geodetic lat[deg], lon[deg], alt[km].
// Returns 0 on success.
int nrlmsis_eval(double utc_jd, double lat_deg, double lon_deg, double alt_km, AtmoOut* out);

// Evaluate with explicit space weather inputs (F10.7a, F10.7, ap[7])
int nrlmsis_eval_with_sw(double utc_jd, double lat_deg, double lon_deg, double alt_km,
                         float f107a, float f107, const std::array<float,7>& ap, AtmoOut* out);

#endif // ATMO_NRLMSIS_H
