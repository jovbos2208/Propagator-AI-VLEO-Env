#include "integrators.h"

namespace vleo_aerodynamics_core {

static void add_scaled(State& y, const State& k, double a) {
    y.r_eci += a * k.r_eci;
    y.v_eci += a * k.v_eci;
    // Quaternion: accumulate in 4-vector space, rewrap later
    Eigen::Vector4d qy(y.q_BI.w(), y.q_BI.x(), y.q_BI.y(), y.q_BI.z());
    Eigen::Vector4d qk(k.q_BI.w(), k.q_BI.x(), k.q_BI.y(), k.q_BI.z());
    qy += a * qk;
    y.q_BI = Eigen::Quaterniond(qy(0), qy(1), qy(2), qy(3));
    y.w_B += a * k.w_B;
}

static State deriv_to_state(const Deriv& d) {
    State s;
    s.r_eci = d.rdot;
    s.v_eci = d.vdot;
    s.q_BI = Eigen::Quaterniond(d.qdot(0), d.qdot(1), d.qdot(2), d.qdot(3));
    s.w_B  = d.wdot;
    return s;
}

void rk4_step(State& x,
              const Controls& u,
              const Params& p,
              const AtmoSample& atmo,
              AeroAdapter& aero,
              double dt) {
    Deriv k1 = dynamics_deriv(x, u, p, atmo, aero);
    State x2 = x; add_scaled(x2, deriv_to_state(k1), dt*0.5); normalize(x2);

    Deriv k2 = dynamics_deriv(x2, u, p, atmo, aero);
    State x3 = x; add_scaled(x3, deriv_to_state(k2), dt*0.5); normalize(x3);

    Deriv k3 = dynamics_deriv(x3, u, p, atmo, aero);
    State x4 = x; add_scaled(x4, deriv_to_state(k3), dt); normalize(x4);

    Deriv k4 = dynamics_deriv(x4, u, p, atmo, aero);

    // Combine
    State dx{};
    add_scaled(dx, deriv_to_state(k1), dt/6.0);
    add_scaled(dx, deriv_to_state(k2), dt/3.0);
    add_scaled(dx, deriv_to_state(k3), dt/3.0);
    add_scaled(dx, deriv_to_state(k4), dt/6.0);

    add_scaled(x, dx, 1.0);
    normalize(x);
}

// Helpers for DP54
static State add_scaled_state(const State& a, const Deriv& d, double s) {
    State r = a;
    r.r_eci += s * d.rdot;
    r.v_eci += s * d.vdot;
    Eigen::Vector4d qv(r.q_BI.w(), r.q_BI.x(), r.q_BI.y(), r.q_BI.z());
    qv += s * d.qdot;
    r.q_BI = Eigen::Quaterniond(qv(0), qv(1), qv(2), qv(3));
    r.w_B  += s * d.wdot;
    return r;
}

static double norm_err(const State& y5, const State& y4, const DP54Params& prm) {
    // Compute a simple weighted RMS error across state components
    double e2 = 0.0;
    auto acc = [&](double val, double scale){ double dv = val/scale; e2 += dv*dv; };
    double s_r = prm.atol + prm.rtol * std::max(y5.r_eci.norm(), 1.0); // scale position
    double s_v = prm.atol + prm.rtol * std::max(y5.v_eci.norm(), 1.0);
    double s_w = prm.atol + prm.rtol * std::max(y5.w_B.norm(), 1.0);
    Eigen::Vector3d dr = y5.r_eci - y4.r_eci;
    Eigen::Vector3d dv = y5.v_eci - y4.v_eci;
    Eigen::Vector3d dw = y5.w_B   - y4.w_B;
    acc(dr.x(), s_r); acc(dr.y(), s_r); acc(dr.z(), s_r);
    acc(dv.x(), s_v); acc(dv.y(), s_v); acc(dv.z(), s_v);
    acc(dw.x(), s_w); acc(dw.y(), s_w); acc(dw.z(), s_w);
    // Quaternion error: difference in 4D
    Eigen::Vector4d q5(y5.q_BI.w(), y5.q_BI.x(), y5.q_BI.y(), y5.q_BI.z());
    Eigen::Vector4d q4(y4.q_BI.w(), y4.q_BI.x(), y4.q_BI.y(), y4.q_BI.z());
    Eigen::Vector4d dq = q5 - q4;
    double s_q = prm.atol + prm.rtol * 1.0;
    for (int i=0;i<4;++i) acc(dq(i), s_q);
    int n = 3+3+3+4; // components counted
    return std::sqrt(e2 / n);
}

bool dp54_adaptive_step(State& x,
                        const Controls& u,
                        const Params& p,
                        const AtmoSample& atmo,
                        AeroAdapter& aero,
                        double& dt,
                        const DP54Params& prm) {
    // Dormand–Prince(5,4) coefficients
    const double c2=1.0/5.0,
                 c3=3.0/10.0,
                 c4=4.0/5.0,
                 c5=8.0/9.0,
                 c6=1.0,
                 c7=1.0;
    const double a21=1.0/5.0;
    const double a31=3.0/40.0, a32=9.0/40.0;
    const double a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0;
    const double a51=19372.0/6561.0, a52=-25360.0/2187.0, a53=64448.0/6561.0, a54=-212.0/729.0;
    const double a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0, a64=49.0/176.0, a65=-5103.0/18656.0;
    const double a71=35.0/384.0, a72=0.0, a73=500.0/1113.0, a74=125.0/192.0, a75=-2187.0/6784.0, a76=11.0/84.0;
    const double b1=35.0/384.0, b2=0.0, b3=500.0/1113.0, b4=125.0/192.0, b5=-2187.0/6784.0, b6=11.0/84.0, b7=0.0;
    const double b1s=5179.0/57600.0, b2s=0.0, b3s=7571.0/16695.0, b4s=393.0/640.0, b5s=-92097.0/339200.0, b6s=187.0/2100.0, b7s=1.0/40.0;

    Deriv k1 = dynamics_deriv(x, u, p, atmo, aero);
    State y2 = add_scaled_state(x, k1, dt*a21); normalize(y2);
    Deriv k2 = dynamics_deriv(y2, u, p, atmo, aero);
    State y3 = add_scaled_state(x, k1, dt*a31); y3 = add_scaled_state(y3, k2, dt*a32); normalize(y3);
    Deriv k3 = dynamics_deriv(y3, u, p, atmo, aero);
    State y4 = add_scaled_state(x, k1, dt*a41); y4 = add_scaled_state(y4, k2, dt*a42); y4 = add_scaled_state(y4, k3, dt*a43); normalize(y4);
    Deriv k4 = dynamics_deriv(y4, u, p, atmo, aero);
    State y5 = add_scaled_state(x, k1, dt*a51); y5 = add_scaled_state(y5, k2, dt*a52); y5 = add_scaled_state(y5, k3, dt*a53); y5 = add_scaled_state(y5, k4, dt*a54); normalize(y5);
    Deriv k5 = dynamics_deriv(y5, u, p, atmo, aero);
    State y6 = add_scaled_state(x, k1, dt*a61); y6 = add_scaled_state(y6, k2, dt*a62); y6 = add_scaled_state(y6, k3, dt*a63); y6 = add_scaled_state(y6, k4, dt*a64); y6 = add_scaled_state(y6, k5, dt*a65); normalize(y6);
    Deriv k6 = dynamics_deriv(y6, u, p, atmo, aero);
    State y7 = add_scaled_state(x, k1, dt*a71); y7 = add_scaled_state(y7, k2, dt*a72); y7 = add_scaled_state(y7, k3, dt*a73); y7 = add_scaled_state(y7, k4, dt*a74); y7 = add_scaled_state(y7, k5, dt*a75); y7 = add_scaled_state(y7, k6, dt*a76); normalize(y7);
    Deriv k7 = dynamics_deriv(y7, u, p, atmo, aero);

    // 5th order solution
    State y5th = x;
    y5th.r_eci += dt * (b1*k1.rdot + b2*k2.rdot + b3*k3.rdot + b4*k4.rdot + b5*k5.rdot + b6*k6.rdot + b7*k7.rdot);
    y5th.v_eci += dt * (b1*k1.vdot + b2*k2.vdot + b3*k3.vdot + b4*k4.vdot + b5*k5.vdot + b6*k6.vdot + b7*k7.vdot);
    {
        Eigen::Vector4d qv(x.q_BI.w(), x.q_BI.x(), x.q_BI.y(), x.q_BI.z());
        qv += dt * (b1*k1.qdot + b2*k2.qdot + b3*k3.qdot + b4*k4.qdot + b5*k5.qdot + b6*k6.qdot + b7*k7.qdot);
        y5th.q_BI = Eigen::Quaterniond(qv(0), qv(1), qv(2), qv(3));
        y5th.q_BI.normalize();
    }
    y5th.w_B  += dt * (b1*k1.wdot + b2*k2.wdot + b3*k3.wdot + b4*k4.wdot + b5*k5.wdot + b6*k6.wdot + b7*k7.wdot);

    // 4th order solution for error estimate
    State y4th = x;
    y4th.r_eci += dt * (b1s*k1.rdot + b2s*k2.rdot + b3s*k3.rdot + b4s*k4.rdot + b5s*k5.rdot + b6s*k6.rdot + b7s*k7.rdot);
    y4th.v_eci += dt * (b1s*k1.vdot + b2s*k2.vdot + b3s*k3.vdot + b4s*k4.vdot + b5s*k5.vdot + b6s*k6.vdot + b7s*k7.vdot);
    {
        Eigen::Vector4d qv(x.q_BI.w(), x.q_BI.x(), x.q_BI.y(), x.q_BI.z());
        qv += dt * (b1s*k1.qdot + b2s*k2.qdot + b3s*k3.qdot + b4s*k4.qdot + b5s*k5.qdot + b6s*k6.qdot + b7s*k7.qdot);
        y4th.q_BI = Eigen::Quaterniond(qv(0), qv(1), qv(2), qv(3));
        y4th.q_BI.normalize();
    }
    y4th.w_B  += dt * (b1s*k1.wdot + b2s*k2.wdot + b3s*k3.wdot + b4s*k4.wdot + b5s*k5.wdot + b6s*k6.wdot + b7s*k7.wdot);

    // Error norm and step control
    double err = norm_err(y5th, y4th, prm);
    if (err <= 1.0) {
        x = y5th;
        normalize(x);
        // Suggest next dt
        double fac = prm.safety * std::pow(std::max(1e-12, err), -0.2); // 1/(order+1) = 1/5
        double dt_new = std::min(prm.max_dt, std::max(prm.min_dt, dt * fac));
        dt = dt_new;
        return true;
    } else {
        double fac = prm.safety * std::pow(std::max(1e-12, err), -0.25); // more conservative on reject
        dt = std::max(prm.min_dt, dt * fac);
        return false;
    }
}

} // namespace vleo_aerodynamics_core
