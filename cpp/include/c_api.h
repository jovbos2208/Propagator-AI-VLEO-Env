#ifndef SHUTTLECOCK_C_API_H
#define SHUTTLECOCK_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct EnvHandle EnvHandle;

typedef struct {
    double eta1_rad;
    double eta2_rad;
    double thrust_N;
} ControlsC;

typedef struct {
    double r_eci[3];
    double v_eci[3];
    double q_BI[4];
    double w_B[3];
    double altitude_m;
    double rho;
    double aoa_rad;
    double aos_rad;
    double roll_rad;
} ObsC;

typedef struct {
    ObsC obs;
    double R_total;
    double R_alt;
    double R_att;
    double R_spin;
    double R_effort;
    double dwell_alt_frac;
    double dwell_att_frac;
    double remaining_impulse_Ns;
    int    substeps;
    int    done; // 0=false, 1=true
} StepResultC;

// Lifecycle
EnvHandle* env_create();
void env_destroy(EnvHandle* h);

// Configuration
// Loads config JSON (flat keys as in env_demo). Returns 0 on success.
int env_load_config(EnvHandle* h, const char* json_path);
void env_set_align_to_velocity(EnvHandle* h, int enabled);
void env_set_integrator(EnvHandle* h, int integrator_type /*0=RK4, 1=DP54*/);
void env_set_debug_csv(EnvHandle* h, const char* path);

// Episode controls
// Returns 0 on success
int env_reset_random(EnvHandle* h, uint64_t seed, double jd0_utc, ObsC* out_obs);

// Stepping
int env_step_duration(EnvHandle* h, ControlsC u, double duration_s, StepResultC* out_step);
int env_step_substeps(EnvHandle* h, ControlsC u, int substeps, StepResultC* out_step);

// Utility: estimate current orbital period [s]
double env_estimate_period_s(EnvHandle* h);

#ifdef __cplusplus
}
#endif

#endif // SHUTTLECOCK_C_API_H
