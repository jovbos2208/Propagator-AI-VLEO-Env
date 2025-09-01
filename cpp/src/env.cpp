#include "env.h"
#include <cmath>
#include <random>
#include <fstream>

namespace vleo_aerodynamics_core {

static Observation build_obs(const State& x, double jd_utc, const AtmoSample& atmo) {
    Observation o;
    o.r_eci = x.r_eci; o.v_eci = x.v_eci; o.q_BI = x.q_BI; o.w_B = x.w_B;
    Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc) * x.r_eci;
    double lat, lon, alt; ecef_to_geodetic(r_ecef, lat, lon, alt);
    o.altitude_m = alt; o.rho = atmo.rho;
    Eigen::Vector3d wind_I = R_ECEF_to_ECI(jd_utc) * atmo.wind_ecef;
    Eigen::Vector3d Vrel_I = x.v_eci - Eigen::Vector3d(0,0,OMEGA_EARTH).cross(x.r_eci) - wind_I;
    Eigen::Vector3d Vrel_B = x.q_BI * Vrel_I; // transform to body
    compute_aoa_aos(Vrel_B, o.aoa_rad, o.aos_rad);
    o.omega_norm_rad_s = x.w_B.norm();
    // Roll about velocity axis: define reference axis from ECI Z projected onto plane ⟂ Vrel_I
    Eigen::Vector3d t = Vrel_I.normalized();
    Eigen::Vector3d ref = Eigen::Vector3d::UnitZ();
    if (std::abs(ref.dot(t)) > 0.95) ref = Eigen::Vector3d::UnitX();
    Eigen::Vector3d u = (ref - ref.dot(t)*t).normalized();
    // Body y-axis in inertial
    Eigen::Matrix3d R_IB = x.q_BI.conjugate().toRotationMatrix();
    Eigen::Vector3d yb_I = R_IB * Eigen::Vector3d::UnitY();
    Eigen::Vector3d v = (yb_I - yb_I.dot(t)*t).normalized();
    double sgn = t.dot(u.cross(v));
    double cval = u.dot(v);
    o.roll_rad = std::atan2(sgn, cval);
    return o;
}

void PropEnv::init(const EnvConfig& cfg) {
    cfg_ = cfg;
    // Geometry and aero init
    aero_.init(cfg_.geometry);
    // Mass/inertia
    params_.mass_inertia.mass_kg = cfg_.mass_props.mass_kg;
    params_.mass_inertia.I_B = cfg_.mass_props.inertia_B;
    params_.mass_inertia.I_B_inv = params_.mass_inertia.I_B.inverse();
}

void PropEnv::set_debug_csv(const std::string& path) {
    if (debug_csv_.is_open()) debug_csv_.close();
    debug_csv_.open(path);
    debug_enabled_ = debug_csv_.good();
    debug_header_written_ = false;
}

Observation PropEnv::reset(double jd0_utc) {
    jd_utc_ = jd0_utc;
    // Initialize state: circular orbit at target altitude in equatorial plane
    const double r0 = params_.gravity.Re + cfg_.init_altitude_m;
    const double v0 = std::sqrt(params_.gravity.mu / r0);
    state_.r_eci = Eigen::Vector3d(r0, 0, 0);
    state_.v_eci = Eigen::Vector3d(0, v0, 0);
    state_.q_BI = Eigen::Quaterniond(1,0,0,0);
    state_.w_B = Eigen::Vector3d::Zero();
    q_ref_ = state_.q_BI;
    remaining_impulse_Ns_ = cfg_.total_impulse_budget_Ns;

    // Initial atmosphere sample
    Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
    double lat, lon, alt_m; ecef_to_geodetic(r_ecef, lat, lon, alt_m);
    AtmoOut out;
    nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m/1000.0,
                         cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
    AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;
    // Optionally align attitude to initial relative velocity
    if (cfg_.align_attitude_to_velocity) {
        Eigen::Vector3d wind_I = R_ECEF_to_ECI(jd_utc_) * atmo.wind_ecef;
        Eigen::Vector3d Vrel_I = state_.v_eci - Eigen::Vector3d(0,0,OMEGA_EARTH).cross(state_.r_eci) - wind_I;
        Eigen::Vector3d t = Vrel_I.normalized();
        Eigen::Vector3d ref = Eigen::Vector3d::UnitZ();
        if (std::abs(ref.dot(t)) > 0.95) ref = Eigen::Vector3d::UnitX();
        Eigen::Vector3d u = (ref - ref.dot(t)*t).normalized();
        Eigen::Vector3d w = t.cross(u);
        Eigen::Matrix3d R_IB; R_IB.col(0)=t; R_IB.col(1)=u; R_IB.col(2)=w;
        Eigen::Matrix3d R_BI = R_IB.transpose();
        state_.q_BI = Eigen::Quaterniond(R_BI);
        state_.q_BI.normalize();
        q_ref_ = state_.q_BI;
    }
    Observation obs0 = build_obs(state_, jd_utc_, atmo);
    roll_ref_rad_ = obs0.roll_rad;
    return obs0;
}

static Eigen::Matrix3d Rz(double a){ double c=std::cos(a), s=std::sin(a); Eigen::Matrix3d R; R<<c,-s,0,s,c,0,0,0,1; return R; }
static Eigen::Matrix3d Rx(double a){ double c=std::cos(a), s=std::sin(a); Eigen::Matrix3d R; R<<1,0,0,0,c,-s,0,s,c; return R; }

static void kepler_to_eci(double a, double e, double inc, double raan, double argp, double nu, double mu,
                          Eigen::Vector3d& r, Eigen::Vector3d& v) {
    const double p = a * (1.0 - e*e);
    const double rmag = p / (1.0 + e * std::cos(nu));
    Eigen::Vector3d r_pqw(rmag * std::cos(nu), rmag * std::sin(nu), 0.0);
    Eigen::Vector3d v_pqw(-std::sqrt(mu/p) * std::sin(nu), std::sqrt(mu/p) * (e + std::cos(nu)), 0.0);
    Eigen::Matrix3d R = Rz(raan) * Rx(inc) * Rz(argp);
    r = R * r_pqw;
    v = R * v_pqw;
}

Observation PropEnv::reset_random(uint64_t seed, double jd0_utc) {
    jd_utc_ = jd0_utc;
    std::mt19937_64 rng(seed);
    auto uni = [&](double a, double b){ std::uniform_real_distribution<double> d(a,b); return d(rng); };

    double alt_m;
    if (cfg_.init_altitude_min_m == cfg_.init_altitude_max_m) alt_m = cfg_.init_altitude_min_m;
    else alt_m = uni(cfg_.init_altitude_min_m, cfg_.init_altitude_max_m);
    double inc = uni(cfg_.init_incl_min_rad, cfg_.init_incl_max_rad);
    double raan = cfg_.randomize_raan ? uni(0.0, 2*M_PI) : cfg_.init_raan_rad;
    double argp = cfg_.randomize_argp ? uni(0.0, 2*M_PI) : cfg_.init_argp_rad;
    double nu   = cfg_.randomize_true_anomaly ? uni(0.0, 2*M_PI) : cfg_.init_true_anomaly_rad;

    double a = params_.gravity.Re + alt_m; // circular
    double e = 0.0;
    kepler_to_eci(a, e, inc, raan, argp, nu, params_.gravity.mu, state_.r_eci, state_.v_eci);
    state_.q_BI = Eigen::Quaterniond(1,0,0,0);
    state_.w_B = Eigen::Vector3d::Zero();
    q_ref_ = state_.q_BI;
    remaining_impulse_Ns_ = cfg_.total_impulse_budget_Ns;

    Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
    double lat, lon, alt_m2; ecef_to_geodetic(r_ecef, lat, lon, alt_m2);
    AtmoOut out;
    nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m2/1000.0,
                         cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
    AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;
    // Optionally align attitude to initial relative velocity
    if (cfg_.align_attitude_to_velocity) {
        Eigen::Vector3d wind_I = R_ECEF_to_ECI(jd_utc_) * atmo.wind_ecef;
        Eigen::Vector3d Vrel_I = state_.v_eci - Eigen::Vector3d(0,0,OMEGA_EARTH).cross(state_.r_eci) - wind_I;
        Eigen::Vector3d t = Vrel_I.normalized();
        Eigen::Vector3d ref = Eigen::Vector3d::UnitZ();
        if (std::abs(ref.dot(t)) > 0.95) ref = Eigen::Vector3d::UnitX();
        Eigen::Vector3d u = (ref - ref.dot(t)*t).normalized();
        Eigen::Vector3d w = t.cross(u);
        Eigen::Matrix3d R_IB; R_IB.col(0)=t; R_IB.col(1)=u; R_IB.col(2)=w;
        Eigen::Matrix3d R_BI = R_IB.transpose();
        state_.q_BI = Eigen::Quaterniond(R_BI);
        state_.q_BI.normalize();
        q_ref_ = state_.q_BI;
    }
    Observation obs0 = build_obs(state_, jd_utc_, atmo);
    roll_ref_rad_ = obs0.roll_rad;
    return obs0;
}

Observation PropEnv::make_observation(const AtmoSample& atmo) const {
    return build_obs(state_, jd_utc_, atmo);
}

static double wrap_pi(double a){ while (a> M_PI) a-=2*M_PI; while (a<-M_PI) a+=2*M_PI; return a; }

RewardTerms PropEnv::compute_reward(const Observation& obs, const Controls& u) const {
    // Tracking terms based on AoA/AoS and altitude
    double alt_err = std::abs(obs.altitude_m - cfg_.target_altitude_m);
    double aoa_err = std::abs(obs.aoa_rad);
    double aos_err = std::abs(obs.aos_rad);
    double roll_err = std::abs(wrap_pi(obs.roll_rad - roll_ref_rad_));
    double effort = std::abs(u.eta1_rad) + std::abs(u.eta2_rad) +
                    (cfg_.thrust_max_N>0? (u.thrust_N/cfg_.thrust_max_N) : 0.0);
    // Spin penalty: sum of absolute angular rates across axes
    double spin_pen = cfg_.w_spin * (std::abs(state_.w_B.x()) + std::abs(state_.w_B.y()) + std::abs(state_.w_B.z()));
    RewardTerms R;
    R.r_alt = -cfg_.w_alt * alt_err;
    R.r_att = -cfg_.w_att_ao * (aoa_err + aos_err) - cfg_.w_att_roll * roll_err;
    R.r_spin = -spin_pen;
    R.r_effort = -cfg_.w_effort * effort;
    R.total = R.r_alt + R.r_att + R.r_spin + R.r_effort;
    return R;
}

bool PropEnv::check_done(const Observation& obs) const {
    if (obs.altitude_m < 80e3) return true;
    if (state_.r_eci.norm() > 20.0 * params_.gravity.Re) return true;
    return false;
}

StepResult PropEnv::step(const Controls& action, double& dt) {
    // Atmosphere sample at current state
    Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
    double lat, lon, alt_m; ecef_to_geodetic(r_ecef, lat, lon, alt_m);
    AtmoOut out;
    nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m/1000.0,
                         cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
    AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;

    // Integrate
    if (integrator_ == IntegratorType::RK4) {
        rk4_step(state_, action, params_, atmo, aero_, dt);
    } else {
        double dt_local = dt;
        int it = 0;
        while (it < 8) {
            double dtn = dt_local;
            bool ok = dp54_adaptive_step(state_, action, params_, atmo, aero_, dtn, dp54_params_);
            if (ok) { dt_local = dtn; break; }
            else { dt_local = dtn; ++it; }
        }
        dt = dt_local;
    }
    jd_utc_ += dt / 86400.0;

    Observation obs = make_observation(atmo);
    RewardTerms R = compute_reward(obs, action);
    bool done = check_done(obs);
    return StepResult{obs, R, done};
}

StepResult PropEnv::step_orbit(const Controls& action) {
    // Estimate orbital period from current state
    const double rmag = state_.r_eci.norm();
    const double vmag2 = state_.v_eci.squaredNorm();
    const double inv_a = 2.0/rmag - vmag2/params_.gravity.mu;
    double a = (inv_a != 0.0) ? (1.0 / inv_a) : (rmag);
    double T = 2.0 * M_PI * std::sqrt(std::max(1e-3, a*a*a) / params_.gravity.mu);
    T = std::min(cfg_.orbit_period_max_s, std::max(cfg_.orbit_period_min_s, T));
    last_T_est_ = T;

    // Integration loop with aggregation
    double t_accum = 0.0;
    double dt = cfg_.dt_initial_s;
    DP54Params dp; dp.atol=cfg_.dp54_atol; dp.rtol=cfg_.dp54_rtol; dp.min_dt=cfg_.dp54_min_dt; dp.max_dt=cfg_.dp54_max_dt; dp.safety=cfg_.dp54_safety;
    RewardTerms Rsum{0,0,0,0};
    double t_in_alt_band = 0.0;
    double t_in_att_band = 0.0;
    Observation obs_end{};
    bool done = false;
    int substeps = 0;
    while (t_accum < T && !done) {
        // Atmosphere sample
        Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
        double lat, lon, alt_m; ecef_to_geodetic(r_ecef, lat, lon, alt_m);
        AtmoOut out;
        nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m/1000.0,
                             cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
        AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;

        if (debug_enabled_) {
            if (!debug_header_written_) {
                debug_csv_ << "jd_utc,alt_m,lat_deg,lon_deg,wx_rad_s,wy_rad_s,wz_rad_s\n";
                debug_header_written_ = true;
            }
            debug_csv_ << jd_utc_ << ','
                       << alt_m << ','
                       << (lat*180.0/M_PI) << ','
                       << (lon*180.0/M_PI) << ','
                       << state_.w_B.x() << ','
                       << state_.w_B.y() << ','
                       << state_.w_B.z() << '\n';
        }

        // Clamp controls
        Controls u = action;
        u.eta1_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta1_rad));
        u.eta2_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta2_rad));
        // Enforce thrust min/max and remaining budget
        double thrust_des = std::max(cfg_.thrust_min_N, std::min(cfg_.thrust_max_N, u.thrust_N));
        double max_by_budget = (dt > 0.0 && remaining_impulse_Ns_ > 0.0) ? (remaining_impulse_Ns_ / dt) : 0.0;
        double thrust_applied = std::min(thrust_des, max_by_budget);
        u.thrust_N = thrust_applied;

        // Integrate one substep
        if (integrator_ == IntegratorType::RK4) {
            rk4_step(state_, u, params_, atmo, aero_, dt);
        } else {
            double dtn = dt;
            bool ok = dp54_adaptive_step(state_, u, params_, atmo, aero_, dtn, dp);
            dt = dtn;
            if (!ok && dt <= dp.min_dt + 1e-12) {
                // Cannot reduce further; break
                done = true;
            }
        }
        jd_utc_ += dt / 86400.0;
        t_accum += dt;
        remaining_impulse_Ns_ = std::max(0.0, remaining_impulse_Ns_ - u.thrust_N * dt);

        Observation obs = build_obs(state_, jd_utc_, atmo);
        RewardTerms Rinst = compute_reward(obs, action);
        Rsum.r_alt += Rinst.r_alt * dt;
        Rsum.r_att += Rinst.r_att * dt;
        Rsum.r_effort += Rinst.r_effort * dt;
        // Dwell counters
        bool in_alt = std::abs(obs.altitude_m - cfg_.target_altitude_m) <= cfg_.alt_band_halfwidth_m;
        bool in_att = (std::abs(obs.aoa_rad) <= cfg_.aoa_band_rad) &&
                      (std::abs(obs.aos_rad) <= cfg_.aos_band_rad) &&
                      (std::abs(wrap_pi(obs.roll_rad - roll_ref_rad_)) <= cfg_.roll_band_rad);
        if (in_alt) t_in_alt_band += dt;
        if (in_att) t_in_att_band += dt;
        obs_end = obs;
        done = done || check_done(obs) || ((cfg_.total_impulse_budget_Ns > 0.0) && (remaining_impulse_Ns_ <= 0.0));
        ++substeps;
    }

    double invT = (last_T_est_ > 0.0) ? (1.0 / last_T_est_) : 0.0;
    RewardTerms Ravg;
    Ravg.r_alt = Rsum.r_alt * invT;
    Ravg.r_att = Rsum.r_att * invT;
    Ravg.r_effort = Rsum.r_effort * invT;
    double dwell_alt_frac = (last_T_est_ > 0.0) ? (t_in_alt_band / last_T_est_) : 0.0;
    double dwell_att_frac = (last_T_est_ > 0.0) ? (t_in_att_band / last_T_est_) : 0.0;
    double dwell_bonus = cfg_.w_dwell_alt * dwell_alt_frac + cfg_.w_dwell_att * dwell_att_frac;
    Ravg.total = Ravg.r_alt + Ravg.r_att + Ravg.r_effort + dwell_bonus;
    StepResult sr; sr.obs = obs_end; sr.rew = Ravg; sr.done = done; sr.dwell_alt_frac = dwell_alt_frac; sr.dwell_att_frac = dwell_att_frac; sr.remaining_impulse_Ns = remaining_impulse_Ns_; sr.substeps = substeps;
    return sr;
}

double PropEnv::estimate_period_s() const {
    const double rmag = state_.r_eci.norm();
    const double vmag2 = state_.v_eci.squaredNorm();
    const double inv_a = 2.0/rmag - vmag2/params_.gravity.mu;
    double a = (inv_a != 0.0) ? (1.0 / inv_a) : (rmag);
    double T = 2.0 * M_PI * std::sqrt(std::max(1e-3, a*a*a) / params_.gravity.mu);
    return std::min(cfg_.orbit_period_max_s, std::max(cfg_.orbit_period_min_s, T));
}

StepResult PropEnv::step_duration(const Controls& action, double duration_s) {
    double T = std::max(0.0, duration_s);
    last_T_est_ = T;

    double t_accum = 0.0;
    double dt = cfg_.dt_initial_s;
    DP54Params dp; dp.atol=cfg_.dp54_atol; dp.rtol=cfg_.dp54_rtol; dp.min_dt=cfg_.dp54_min_dt; dp.max_dt=cfg_.dp54_max_dt; dp.safety=cfg_.dp54_safety;
    RewardTerms Rsum{0,0,0,0,0};
    double t_in_alt_band = 0.0;
    double t_in_att_band = 0.0;
    Observation obs_end{};
    bool done = false;
    int substeps = 0;
    while (t_accum < T && !done) {
        // Atmosphere sample
        Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
        double lat, lon, alt_m; ecef_to_geodetic(r_ecef, lat, lon, alt_m);
        AtmoOut out;
        nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m/1000.0,
                             cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
        AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;

        if (debug_enabled_) {
            if (!debug_header_written_) { debug_csv_ << "jd_utc,alt_m,lat_deg,lon_deg,wx_rad_s,wy_rad_s,wz_rad_s\n"; debug_header_written_ = true; }
            debug_csv_ << jd_utc_ << ',' << alt_m << ',' << (lat*180.0/M_PI) << ',' << (lon*180.0/M_PI) << ','
                       << state_.w_B.x() << ',' << state_.w_B.y() << ',' << state_.w_B.z() << '\n';
        }

        // Clamp controls and budget
        Controls u = action;
        u.eta1_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta1_rad));
        u.eta2_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta2_rad));
        double thrust_des = std::max(cfg_.thrust_min_N, std::min(cfg_.thrust_max_N, u.thrust_N));
        double max_by_budget = (dt > 0.0 && cfg_.total_impulse_budget_Ns>0.0 && remaining_impulse_Ns_ > 0.0) ? (remaining_impulse_Ns_ / dt) : thrust_des;
        double thrust_applied = std::min(thrust_des, max_by_budget);
        u.thrust_N = thrust_applied;

        // Integrate one substep
        if (integrator_ == IntegratorType::RK4) { rk4_step(state_, u, params_, atmo, aero_, dt); }
        else { double dtn = dt; bool ok = dp54_adaptive_step(state_, u, params_, atmo, aero_, dtn, dp); dt = dtn; if (!ok && dt <= dp.min_dt + 1e-12) done = true; }
        jd_utc_ += dt / 86400.0; t_accum += dt; if (cfg_.total_impulse_budget_Ns>0.0) remaining_impulse_Ns_ = std::max(0.0, remaining_impulse_Ns_ - u.thrust_N * dt);

        Observation obs = build_obs(state_, jd_utc_, atmo);
        RewardTerms Rinst = compute_reward(obs, action);
        Rsum.r_alt += Rinst.r_alt * dt; Rsum.r_att += Rinst.r_att * dt; Rsum.r_spin += Rinst.r_spin * dt; Rsum.r_effort += Rinst.r_effort * dt;
        bool in_alt = std::abs(obs.altitude_m - cfg_.target_altitude_m) <= cfg_.alt_band_halfwidth_m;
        bool in_att = (std::abs(obs.aoa_rad) <= cfg_.aoa_band_rad) && (std::abs(obs.aos_rad) <= cfg_.aos_band_rad) && (std::abs(wrap_pi(obs.roll_rad - roll_ref_rad_)) <= cfg_.roll_band_rad);
        if (in_alt) t_in_alt_band += dt; if (in_att) t_in_att_band += dt;
        obs_end = obs; done = done || check_done(obs) || ((cfg_.total_impulse_budget_Ns > 0.0) && (remaining_impulse_Ns_ <= 0.0)); ++substeps;
    }

    double invT = (last_T_est_ > 0.0) ? (1.0 / last_T_est_) : 0.0;
    RewardTerms Ravg; Ravg.r_alt = Rsum.r_alt * invT; Ravg.r_att = Rsum.r_att * invT; Ravg.r_spin = Rsum.r_spin * invT; Ravg.r_effort = Rsum.r_effort * invT; Ravg.total = Ravg.r_alt + Ravg.r_att + Ravg.r_spin + Ravg.r_effort + (cfg_.w_dwell_alt * (t_in_alt_band*invT) + cfg_.w_dwell_att * (t_in_att_band*invT));
    StepResult sr; sr.obs = obs_end; sr.rew = Ravg; sr.done = done; sr.dwell_alt_frac = (t_in_alt_band*invT); sr.dwell_att_frac = (t_in_att_band*invT); sr.remaining_impulse_Ns = remaining_impulse_Ns_; sr.substeps = substeps; return sr;
}

StepResult PropEnv::step_substeps(const Controls& action, int substeps_target) {
    int target = std::max(1, substeps_target);
    double t_accum = 0.0;
    double dt = cfg_.dt_initial_s;
    DP54Params dp; dp.atol=cfg_.dp54_atol; dp.rtol=cfg_.dp54_rtol; dp.min_dt=cfg_.dp54_min_dt; dp.max_dt=cfg_.dp54_max_dt; dp.safety=cfg_.dp54_safety;
    RewardTerms Rsum{0,0,0,0,0};
    double t_in_alt_band = 0.0;
    double t_in_att_band = 0.0;
    Observation obs_end{};
    bool done = false;
    int substeps = 0;
    while (substeps < target && !done) {
        // Atmosphere
        Eigen::Vector3d r_ecef = R_ECI_to_ECEF(jd_utc_) * state_.r_eci;
        double lat, lon, alt_m; ecef_to_geodetic(r_ecef, lat, lon, alt_m);
        AtmoOut out;
        nrlmsis_eval_with_sw(jd_utc_, lat*180.0/M_PI, lon*180.0/M_PI, alt_m/1000.0,
                             cfg_.space_weather.f107a, cfg_.space_weather.f107, cfg_.space_weather.ap, &out);
        AtmoSample atmo; atmo.jd_utc = jd_utc_; atmo.rho = out.rho_kg_m3; atmo.T_K = out.T_K; atmo.wind_ecef = out.wind_ecef; atmo.particles_mass_kg = out.mean_molecular_mass_kg;

        if (debug_enabled_) {
            if (!debug_header_written_) { debug_csv_ << "jd_utc,alt_m,lat_deg,lon_deg,wx_rad_s,wy_rad_s,wz_rad_s\n"; debug_header_written_ = true; }
            debug_csv_ << jd_utc_ << ',' << alt_m << ',' << (lat*180.0/M_PI) << ',' << (lon*180.0/M_PI) << ','
                       << state_.w_B.x() << ',' << state_.w_B.y() << ',' << state_.w_B.z() << '\n';
        }

        // Clamp controls and budget
        Controls u = action;
        u.eta1_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta1_rad));
        u.eta2_rad = std::max(-cfg_.eta_limit_rad, std::min(cfg_.eta_limit_rad, u.eta2_rad));
        double thrust_des = std::max(cfg_.thrust_min_N, std::min(cfg_.thrust_max_N, u.thrust_N));
        double max_by_budget = (dt > 0.0 && cfg_.total_impulse_budget_Ns>0.0 && remaining_impulse_Ns_ > 0.0) ? (remaining_impulse_Ns_ / dt) : thrust_des;
        double thrust_applied = std::min(thrust_des, max_by_budget);
        u.thrust_N = thrust_applied;

        // One integrator substep
        if (integrator_ == IntegratorType::RK4) { rk4_step(state_, u, params_, atmo, aero_, dt); }
        else { double dtn = dt; bool ok = dp54_adaptive_step(state_, u, params_, atmo, aero_, dtn, dp); dt = dtn; if (!ok && dt <= dp.min_dt + 1e-12) done = true; }
        jd_utc_ += dt / 86400.0; t_accum += dt; if (cfg_.total_impulse_budget_Ns>0.0) remaining_impulse_Ns_ = std::max(0.0, remaining_impulse_Ns_ - u.thrust_N * dt);

        Observation obs = build_obs(state_, jd_utc_, atmo);
        RewardTerms Rinst = compute_reward(obs, action);
        Rsum.r_alt += Rinst.r_alt * dt; Rsum.r_att += Rinst.r_att * dt; Rsum.r_spin += Rinst.r_spin * dt; Rsum.r_effort += Rinst.r_effort * dt;
        bool in_alt = std::abs(obs.altitude_m - cfg_.target_altitude_m) <= cfg_.alt_band_halfwidth_m;
        bool in_att = (std::abs(obs.aoa_rad) <= cfg_.aoa_band_rad) && (std::abs(obs.aos_rad) <= cfg_.aos_band_rad) && (std::abs(wrap_pi(obs.roll_rad - roll_ref_rad_)) <= cfg_.roll_band_rad);
        if (in_alt) t_in_alt_band += dt; if (in_att) t_in_att_band += dt;
        obs_end = obs; done = done || check_done(obs) || ((cfg_.total_impulse_budget_Ns > 0.0) && (remaining_impulse_Ns_ <= 0.0)); ++substeps;
    }

    last_T_est_ = t_accum;
    double invT = (last_T_est_ > 0.0) ? (1.0 / last_T_est_) : 0.0;
    RewardTerms Ravg; Ravg.r_alt = Rsum.r_alt * invT; Ravg.r_att = Rsum.r_att * invT; Ravg.r_spin = Rsum.r_spin * invT; Ravg.r_effort = Rsum.r_effort * invT; Ravg.total = Ravg.r_alt + Ravg.r_att + Ravg.r_spin + Ravg.r_effort + (cfg_.w_dwell_alt * (t_in_alt_band*invT) + cfg_.w_dwell_att * (t_in_att_band*invT));
    StepResult sr; sr.obs = obs_end; sr.rew = Ravg; sr.done = done; sr.dwell_alt_frac = (t_in_alt_band*invT); sr.dwell_att_frac = (t_in_att_band*invT); sr.remaining_impulse_Ns = remaining_impulse_Ns_; sr.substeps = substeps; return sr;
}

} // namespace vleo_aerodynamics_core
