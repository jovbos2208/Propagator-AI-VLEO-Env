#include "c_api.h"
#include "env.h"
#include <memory>
#include <string>
#include <fstream>
#include <sstream>

using namespace vleo_aerodynamics_core;

struct EnvHandle {
    PropEnv env;
};

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

static bool find_number(const std::string& s, const std::string& key, double& out) {
    auto pos = s.find("\""+key+"\""); if (pos==std::string::npos) return false;
    pos = s.find(':', pos); if (pos==std::string::npos) return false;
    auto end = s.find_first_of(",}\n", pos+1);
    std::string token = s.substr(pos+1, end-pos-1);
    try { out = std::stod(token); return true; } catch(...) { return false; }
}

static bool find_string(const std::string& s, const std::string& key, std::string& out) {
    auto pos = s.find("\""+key+"\""); if (pos==std::string::npos) return false;
    pos = s.find(':', pos); if (pos==std::string::npos) return false;
    pos = s.find('"', pos+1); if (pos==std::string::npos) return false;
    auto end = s.find('"', pos+1); if (end==std::string::npos) return false;
    out = s.substr(pos+1, end-pos-1); return true;
}

static bool find_array7(const std::string& s, const std::string& key, std::array<float,7>& out) {
    auto pos = s.find("\""+key+"\""); if (pos==std::string::npos) return false;
    pos = s.find('[', pos); if (pos==std::string::npos) return false;
    auto end = s.find(']', pos+1); if (end==std::string::npos) return false;
    std::string token = s.substr(pos+1, end-pos-1);
    std::replace(token.begin(), token.end(), ',', ' ');
    std::istringstream iss(token);
    for (int i=0;i<7;++i) { if (!(iss>>out[i])) return false; }
    return true;
}

EnvHandle* env_create() {
    auto* h = new EnvHandle();
    // default config
    EnvConfig cfg;
    h->env.init(cfg);
    return h;
}

void env_destroy(EnvHandle* h) {
    delete h;
}

int env_load_config(EnvHandle* h, const char* json_path) {
    if (!h || !json_path) return -1;
    std::string js = slurp(json_path);
    if (js.empty()) return -2;
    EnvConfig cfg;
    double tmp;
    std::string integ;
    if (find_string(js, "integrator", integ)) {
        if (integ == "rk4") h->env.set_integrator(IntegratorType::RK4);
        else h->env.set_integrator(IntegratorType::DP54);
    }
    if (find_number(js, "dt", tmp)) cfg.dt_initial_s = tmp;
    if (find_number(js, "alt_min", tmp)) cfg.init_altitude_min_m = tmp;
    if (find_number(js, "alt_max", tmp)) cfg.init_altitude_max_m = tmp;
    if (find_number(js, "incl_min_deg", tmp)) cfg.init_incl_min_rad = tmp * M_PI/180.0;
    if (find_number(js, "incl_max_deg", tmp)) cfg.init_incl_max_rad = tmp * M_PI/180.0;
    if (find_number(js, "target_altitude", tmp)) cfg.target_altitude_m = tmp;
    if (find_number(js, "eta_limit_deg", tmp)) cfg.eta_limit_rad = tmp * M_PI/180.0;
    if (find_number(js, "thrust_min", tmp)) cfg.thrust_min_N = tmp;
    if (find_number(js, "thrust_max", tmp)) cfg.thrust_max_N = tmp;
    if (find_number(js, "impulse_budget", tmp)) cfg.total_impulse_budget_Ns = tmp;
    if (find_number(js, "w_alt", tmp)) cfg.w_alt = tmp;
    if (find_number(js, "w_ao", tmp)) cfg.w_att_ao = tmp;
    if (find_number(js, "w_roll", tmp)) cfg.w_att_roll = tmp;
    if (find_number(js, "w_effort", tmp)) cfg.w_effort = tmp;
    if (find_number(js, "w_dwell_alt", tmp)) cfg.w_dwell_alt = tmp;
    if (find_number(js, "w_dwell_att", tmp)) cfg.w_dwell_att = tmp;
    if (find_number(js, "alt_band", tmp)) cfg.alt_band_halfwidth_m = tmp;
    if (find_number(js, "aoa_band_deg", tmp)) cfg.aoa_band_rad = tmp * M_PI/180.0;
    if (find_number(js, "aos_band_deg", tmp)) cfg.aos_band_rad = tmp * M_PI/180.0;
    if (find_number(js, "roll_band_deg", tmp)) cfg.roll_band_rad = tmp * M_PI/180.0;
    if (find_number(js, "omega_band_deg_s", tmp)) cfg.omega_band_rad_s = tmp * M_PI/180.0;
    if (find_number(js, "f107a", tmp)) cfg.space_weather.f107a = (float)tmp;
    if (find_number(js, "f107", tmp)) cfg.space_weather.f107 = (float)tmp;
    find_array7(js, "ap", cfg.space_weather.ap);
    // Optional atmosphere cache controls
    if (find_number(js, "atmo_cache_period_s", tmp)) cfg.atmo_cache_period_s = tmp;
    if (find_number(js, "atmo_cache_alt_tol_m", tmp)) cfg.atmo_cache_alt_tol_m = tmp;
    if (find_number(js, "atmo_cache_latlon_tol_deg", tmp)) cfg.atmo_cache_latlon_tol_deg = tmp;

    // Optional geometry overrides via JSON (file paths)
    // Keys (strings): geometry_dir, main_body_obj, wing_right_obj, wing_top_obj, wing_left_obj, wing_bottom_obj
    std::string geom_dir;
    find_string(js, "geometry_dir", geom_dir);
    auto make_path = [&](const std::string& p){
        if (p.empty()) return p;
        // Prepend geometry_dir for relative paths
        if (!geom_dir.empty()) {
            if (!(p.size() > 0 && (p[0] == '/'
#ifdef _WIN32
                  || (p.size() > 1 && p[1] == ':')
#endif
                ))) {
                return geom_dir + "/" + p;
            }
        }
        return p;
    };
    std::string f0, f1, f2, f3, f4;
    bool any_geom = false;
    if (find_string(js, "main_body_obj", f0)) { cfg.geometry.object_files[0] = make_path(f0); any_geom = true; }
    if (find_string(js, "wing_right_obj", f1)) { cfg.geometry.object_files[1] = make_path(f1); any_geom = true; }
    if (find_string(js, "wing_top_obj", f2)) { cfg.geometry.object_files[2] = make_path(f2); any_geom = true; }
    if (find_string(js, "wing_left_obj", f3)) { cfg.geometry.object_files[3] = make_path(f3); any_geom = true; }
    if (find_string(js, "wing_bottom_obj", f4)) { cfg.geometry.object_files[4] = make_path(f4); any_geom = true; }
    h->env.init(cfg);
    return 0;
}

void env_set_align_to_velocity(EnvHandle* h, int enabled) {
    if (!h) return;
    h->env.cfg_.align_attitude_to_velocity = (enabled != 0);
}

void env_set_integrator(EnvHandle* h, int t) {
    if (!h) return;
    if (t == 0) h->env.set_integrator(IntegratorType::RK4);
    else h->env.set_integrator(IntegratorType::DP54);
}

void env_set_debug_csv(EnvHandle* h, const char* path) {
    if (!h || !path) return;
    h->env.set_debug_csv(path);
}

static void copy_obs(const vleo_aerodynamics_core::Observation& o, ObsC* out) {
    for (int i=0;i<3;++i) { out->r_eci[i]=o.r_eci(i); out->v_eci[i]=o.v_eci(i); out->w_B[i]=o.w_B(i); }
    out->q_BI[0]=o.q_BI.w(); out->q_BI[1]=o.q_BI.x(); out->q_BI[2]=o.q_BI.y(); out->q_BI[3]=o.q_BI.z();
    out->altitude_m=o.altitude_m; out->rho=o.rho; out->aoa_rad=o.aoa_rad; out->aos_rad=o.aos_rad; out->roll_rad=o.roll_rad;
}

int env_reset_random(EnvHandle* h, uint64_t seed, double jd0_utc, ObsC* out_obs) {
    if (!h || !out_obs) return -1;
    auto obs = h->env.reset_random(seed, jd0_utc);
    copy_obs(obs, out_obs);
    return 0;
}

static Controls to_controls(const ControlsC& uc) {
    Controls u; u.eta1_rad=uc.eta1_rad; u.eta2_rad=uc.eta2_rad; u.thrust_N=uc.thrust_N; return u;
}

static void copy_step(const StepResult& sr, StepResultC* out) {
    copy_obs(sr.obs, &out->obs);
    out->R_total=sr.rew.total; out->R_alt=sr.rew.r_alt; out->R_att=sr.rew.r_att; out->R_spin=sr.rew.r_spin; out->R_effort=sr.rew.r_effort;
    out->dwell_alt_frac=sr.dwell_alt_frac; out->dwell_att_frac=sr.dwell_att_frac; out->remaining_impulse_Ns=sr.remaining_impulse_Ns; out->substeps=sr.substeps; out->done=sr.done?1:0;
}

int env_step_duration(EnvHandle* h, ControlsC uc, double duration_s, StepResultC* out_step) {
    if (!h || !out_step) return -1;
    StepResult sr = h->env.step_duration(to_controls(uc), duration_s);
    copy_step(sr, out_step);
    return 0;
}

int env_step_substeps(EnvHandle* h, ControlsC uc, int substeps, StepResultC* out_step) {
    if (!h || !out_step) return -1;
    StepResult sr = h->env.step_substeps(to_controls(uc), substeps);
    copy_step(sr, out_step);
    return 0;
}

double env_estimate_period_s(EnvHandle* h) {
    if (!h) return 0.0;
    return h->env.estimate_period_s();
}
