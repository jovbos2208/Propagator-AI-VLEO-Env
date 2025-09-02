#ifndef PROP_ENV_H
#define PROP_ENV_H

#include "env_config.h"
#include "dynamics.h"
#include "integrators.h"
#include "frames.h"
#include "atmo_nrlmsis.h"
#include <fstream>

namespace vleo_aerodynamics_core {

enum class IntegratorType { RK4, DP54 };

struct Observation {
    Eigen::Vector3d r_eci;
    Eigen::Vector3d v_eci;
    Eigen::Quaterniond q_BI;
    Eigen::Vector3d w_B;
    double altitude_m;
    double rho;
    double aoa_rad;
    double aos_rad;
    double omega_norm_rad_s; // for roll/spin proxy
    double roll_rad;         // roll about velocity axis relative to inertial reference
};

struct RewardTerms {
    double r_alt;
    double r_att;
    double r_spin;
    double r_effort;
    double total;
};

struct StepResult {
    Observation obs;
    RewardTerms rew;
    bool done;
    double dwell_alt_frac = 0.0;
    double dwell_att_frac = 0.0;
    double remaining_impulse_Ns = 0.0;
    int substeps = 0; // number of integration substeps used this orbit
};

class PropEnv {
public:
    void init(const EnvConfig& cfg);
    Observation reset(double jd0_utc);
    Observation reset_random(uint64_t seed, double jd0_utc);
    StepResult step(const Controls& action, double& dt);
    StepResult step_orbit(const Controls& action);
    StepResult step_duration(const Controls& action, double duration_s);
    StepResult step_substeps(const Controls& action, int substeps_target);

    void set_integrator(IntegratorType t, const DP54Params& dp = DP54Params{}) {
        integrator_ = t; dp54_params_ = dp;
    }

    // Debug per-substep CSV logging
    void set_debug_csv(const std::string& path);

    // Utility: estimate current orbital period from instantaneous state
    double estimate_period_s() const;

public:
    EnvConfig cfg_;
    Params params_;
    AeroAdapter aero_;
    State state_;
    double jd_utc_ = 2451545.0;
    Eigen::Quaterniond q_ref_ = Eigen::Quaterniond(1,0,0,0);
    double roll_ref_rad_ = 0.0;
    double last_T_est_ = 0.0;
    double remaining_impulse_Ns_ = 0.0;

    // Debug logging
    bool debug_enabled_ = false;
    std::ofstream debug_csv_;
    bool debug_header_written_ = false;

private:
    Observation make_observation(const AtmoSample& atmo) const;
    RewardTerms compute_reward(const Observation& obs, const Controls& u) const;
    bool check_done(const Observation& obs) const;

    // Atmosphere sampling with optional caching
    AtmoOut sample_atmo_cached(double jd_utc, double lat_deg, double lon_deg, double alt_km);


    IntegratorType integrator_ = IntegratorType::RK4;
    DP54Params dp54_params_{};

    // Cache state for atmosphere
    AtmoOut last_atmo_{};
    double last_atmo_jd_ = -1.0;
    double last_lat_deg_ = 1e9;
    double last_lon_deg_ = 1e9;
    double last_alt_m_ = 1e9;
};

} // namespace vleo_aerodynamics_core

#endif // PROP_ENV_H
