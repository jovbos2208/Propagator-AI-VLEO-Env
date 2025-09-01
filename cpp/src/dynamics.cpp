#include "dynamics.h"
#include <cmath>

namespace vleo_aerodynamics_core {

static Eigen::Matrix3d R_BI_from_quat(const Eigen::Quaterniond& q_BI) {
    return q_BI.toRotationMatrix();
}

static Eigen::Matrix3d R_IB_from_quat(const Eigen::Quaterniond& q_BI) {
    return q_BI.conjugate().toRotationMatrix();
}

static Eigen::Vector4d quat_deriv(const Eigen::Quaterniond& q, const Eigen::Vector3d& w_B) {
    // qdot = 0.5 * Omega(w) * q, with Omega in body frame
    Eigen::Matrix4d Omega;
    Omega <<  0,      -w_B.x(), -w_B.y(), -w_B.z(),
              w_B.x(), 0,        w_B.z(), -w_B.y(),
              w_B.y(), -w_B.z(), 0,        w_B.x(),
              w_B.z(), w_B.y(), -w_B.x(),  0;
    Eigen::Vector4d qv(q.w(), q.x(), q.y(), q.z());
    return 0.5 * (Omega * qv);
}

void normalize(State& x) {
    x.q_BI.normalize();
}

void compute_aoa_aos(const Eigen::Vector3d& Vrel_B, double& aoa_rad, double& aos_rad) {
    const double Vx = Vrel_B.x();
    const double Vy = Vrel_B.y();
    const double Vz = Vrel_B.z();
    aoa_rad = std::atan2(-Vz, Vx);
    aos_rad = std::atan2(Vy, std::sqrt(Vx*Vx + Vz*Vz));
}

Deriv dynamics_deriv(const State& x,
                     const Controls& u,
                     const Params& p,
                     const AtmoSample& atmo,
                     AeroAdapter& aero) {
    Deriv d;
    d.rdot = x.v_eci;

    // Frames
    Eigen::Matrix3d R_EI = R_ECEF_to_ECI(atmo.jd_utc); // ECEF->ECI
    Eigen::Matrix3d R_IE = R_ECI_to_ECEF(atmo.jd_utc); // ECI->ECEF
    Eigen::Matrix3d R_BI = R_BI_from_quat(x.q_BI);
    Eigen::Matrix3d R_IB = R_IB_from_quat(x.q_BI);

    // Winds and Vrel in inertial
    Eigen::Vector3d wind_I = R_EI * atmo.wind_ecef; // map wind to ECI
    Eigen::Vector3d Vrel_I = x.v_eci - Eigen::Vector3d(0,0,OMEGA_EARTH).cross(x.r_eci) - wind_I;

    // Aero F/T in body frame (black-box)
    auto [F_B, T_B] = aero.computeFT(x.q_BI, x.w_B, x.v_eci, wind_I,
                                     atmo.rho, atmo.T_K, atmo.particles_mass_kg,
                                     u.eta1_rad, u.eta2_rad, /*temp_ratio_method=*/1);

    // Thrust along +X_B
    Eigen::Vector3d F_thrust_B(u.thrust_N, 0.0, 0.0);

    // Gravity in ECI
    Eigen::Vector3d a_grav = gravity_accel_eci(x.r_eci, p.gravity);

    // Sum accelerations in ECI
    Eigen::Vector3d F_total_B = F_B + F_thrust_B;
    Eigen::Vector3d a_aero_thrust_I = R_IB * F_total_B / p.mass_inertia.mass_kg;
    d.vdot = a_grav + a_aero_thrust_I;

    // Attitude dynamics
    Eigen::Vector4d qdot = quat_deriv(x.q_BI, x.w_B);
    d.qdot = qdot;

    Eigen::Vector3d T_total_B = T_B; // no thrust torque by default
    Eigen::Vector3d Iw = p.mass_inertia.I_B * x.w_B;
    d.wdot = p.mass_inertia.I_B_inv * (T_total_B - x.w_B.cross(Iw));

    return d;
}

} // namespace vleo_aerodynamics_core

