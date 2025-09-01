#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <Eigen/Dense>
#include "frames.h"
#include "gravity.h"
#include "aero_adapter.h"

namespace vleo_aerodynamics_core {

struct State {
    Eigen::Vector3d r_eci;   // position [m]
    Eigen::Vector3d v_eci;   // velocity [m/s]
    Eigen::Quaterniond q_BI; // body <- inertial
    Eigen::Vector3d w_B;     // angular velocity in body [rad/s]
};

struct Controls {
    double eta1_rad = 0.0;
    double eta2_rad = 0.0;
    double thrust_N = 0.0; // along +X_B
};

struct MassInertia {
    double mass_kg = 1.0;
    Eigen::Matrix3d I_B = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d I_B_inv = Eigen::Matrix3d::Identity();
};

struct AtmoSample {
    double jd_utc = 2451545.0; // time for transforms (UTC→UT1 approx)
    double rho = 0.0;          // kg/m^3
    double T_K = 0.0;          // K
    Eigen::Vector3d wind_ecef = Eigen::Vector3d::Zero(); // m/s
    double particles_mass_kg = 16.0 * 1.6605390689252e-27; // kg
};

struct Params {
    GravityParams gravity;
    MassInertia mass_inertia;
};

struct Deriv {
    Eigen::Vector3d rdot;
    Eigen::Vector3d vdot;
    Eigen::Vector4d qdot; // [w x y z]
    Eigen::Vector3d wdot;
};

// Compute derivatives using aero black-box and gravity
Deriv dynamics_deriv(const State& x,
                     const Controls& u,
                     const Params& p,
                     const AtmoSample& atmo,
                     AeroAdapter& aero);

// Utility to normalize quaternion inside state
void normalize(State& x);

// AoA/AoS computation from Vrel_B
void compute_aoa_aos(const Eigen::Vector3d& Vrel_B, double& aoa_rad, double& aos_rad);

} // namespace vleo_aerodynamics_core

#endif // DYNAMICS_H

