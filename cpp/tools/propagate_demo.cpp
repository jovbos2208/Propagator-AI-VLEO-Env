#include "env_config.h"
#include "aero_adapter.h"
#include "frames.h"
#include "gravity.h"
#include "dynamics.h"
#include "integrators.h"
#include <Eigen/Dense>
#include "atmo_nrlmsis.h"
#include <iostream>
#include <cmath>

using namespace vleo_aerodynamics_core;

int main() {
    // Initialize geometry (five-part shuttlecock)
    GeometryConfig geom = ShuttlecockDefaults::geometry();
    AeroAdapter aero;
    try { aero.init(geom); }
    catch (const std::exception& e) { std::cerr << "Geometry init error: " << e.what() << "\n"; return 1; }

    // Parameters
    Params params;
    params.mass_inertia.mass_kg = 5.0; // example mass
    params.mass_inertia.I_B << 0.05,0,0, 0,0.05,0, 0,0,0.05; // rough inertia
    params.mass_inertia.I_B_inv = params.mass_inertia.I_B.inverse();

    // Initial circular orbit at 300 km altitude in equator plane
    const double alt0 = 300e3;
    const double r0 = params.gravity.Re + alt0;
    const double v0 = std::sqrt(params.gravity.mu / r0);
    State x;
    x.r_eci = Eigen::Vector3d(r0, 0, 0);
    x.v_eci = Eigen::Vector3d(0, v0, 0);
    x.q_BI = Eigen::Quaterniond(1,0,0,0);
    x.w_B = Eigen::Vector3d::Zero();

    Controls u; // zero thrust, zero deflection
    AtmoSample atmo;
    atmo.jd_utc = 2451545.0; // J2000 as placeholder

    const double dt = 0.5; // s
    const int steps = 2000; // ~1000 s

    for (int i = 0; i < steps; ++i) {
        // Query atmosphere via NRLMSIS
        Eigen::Vector3d r_ecef = R_ECI_to_ECEF(atmo.jd_utc) * x.r_eci;
        double lat, lon, alt_m;
        ecef_to_geodetic(r_ecef, lat, lon, alt_m);
        AtmoOut out;
        nrlmsis_eval(atmo.jd_utc, lat * 180.0/M_PI, lon * 180.0/M_PI, alt_m/1000.0, &out);
        atmo.rho = out.rho_kg_m3;
        atmo.T_K = out.T_K;
        atmo.wind_ecef = out.wind_ecef; // zero for now
        atmo.particles_mass_kg = out.mean_molecular_mass_kg;

        rk4_step(x, u, params, atmo, aero, dt);
        atmo.jd_utc += dt / 86400.0; // advance time

        if ((i % 100) == 0) {
            double speed = x.v_eci.norm();
            std::cout << "t=" << i*dt << " s, alt=" << alt_m << " m, rho=" << atmo.rho
                      << " kg/m^3, T=" << atmo.T_K << " K, |v|=" << speed << " m/s\n";
        }
    }

    std::cout << "Demo complete." << std::endl;
    return 0;
}
