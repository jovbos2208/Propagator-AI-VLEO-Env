#include "env.h"
#include "env_config.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <fstream>
#include <cstring>

using namespace vleo_aerodynamics_core;

static double deg2rad(double d){ return d*M_PI/180.0; }

struct Args {
    int episodes = 1;
    int orbits = 5;
    std::string integrator = "dp54";
    uint64_t seed = 1234;
    double dt = 1.0;
    double alt_min = 300e3, alt_max = 300e3;
    double incl_min_deg = 0.0, incl_max_deg = 0.0;
    double target_alt = 300e3;
    double eta_limit_deg = 45.0;
    double thrust_min = 0.0, thrust_max = 0.0, impulse_budget = 0.0;
    double w_alt = 1e-6, w_ao = 1.0, w_roll = 1.0, w_effort = 1e-2, w_dwell_alt=1.0, w_dwell_att=1.0;
    double alt_band = 1000.0;
    double aoa_band_deg = 5.0, aos_band_deg = 5.0, roll_band_deg = 5.0, omega_band_deg_s = 5.0;
    float f107a = 150.0f, f107 = 150.0f; std::array<float,7> ap = {15,12,12,12,12,12,12};
    std::string config_path;
    std::string log_csv_path;
    bool align_to_velocity = false;
    std::string debug_csv_path;
    double control_dt = 0.0;
    int control_substeps = 0;
};

static void print_usage() {
    std::cout
        << "env_demo [--episodes N] [--orbits N] [--integrator rk4|dp54] [--seed S]\n"
        << "         [--dt DT] [--alt-min m] [--alt-max m] [--incl-min deg] [--incl-max deg]\n"
        << "         [--target-alt m] [--eta-limit deg] [--thrust-min N] [--thrust-max N] [--impulse-budget Ns]\n"
        << "         [--w-alt v] [--w-ao v] [--w-roll v] [--w-effort v] [--w-dwell-alt v] [--w-dwell-att v]\n"
        << "         [--alt-band m] [--aoa-band deg] [--aos-band deg] [--roll-band deg] [--omega-band deg_s]\n"
        << "         [--f107a v] [--f107 v] [--ap a,b,c,d,e,f,g]\n"
        << "         [--config path.json] [--log-csv path.csv] [--align-to-velocity] [--debug-csv path.csv]\n"
        << "         [--control-dt seconds]  (act every control-dt instead of per-orbit)\n"
        << "         [--control-substeps N] (act every N integrator substeps)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    auto need = [&](int i){ if (i+1>=argc){ std::cerr<<"Missing value for "<<argv[i]<<"\n"; return false;} return true; };
    for (int i=1;i<argc;++i){ std::string k=argv[i];
        if (k=="--episodes" && need(i)) a.episodes = std::stoi(argv[++i]);
        else if (k=="--orbits" && need(i)) a.orbits = std::stoi(argv[++i]);
        else if (k=="--integrator" && need(i)) a.integrator = argv[++i];
        else if (k=="--seed" && need(i)) a.seed = std::stoull(argv[++i]);
        else if (k=="--dt" && need(i)) a.dt = std::stod(argv[++i]);
        else if (k=="--alt-min" && need(i)) a.alt_min = std::stod(argv[++i]);
        else if (k=="--alt-max" && need(i)) a.alt_max = std::stod(argv[++i]);
        else if (k=="--incl-min" && need(i)) a.incl_min_deg = std::stod(argv[++i]);
        else if (k=="--incl-max" && need(i)) a.incl_max_deg = std::stod(argv[++i]);
        else if (k=="--target-alt" && need(i)) a.target_alt = std::stod(argv[++i]);
        else if (k=="--eta-limit" && need(i)) a.eta_limit_deg = std::stod(argv[++i]);
        else if (k=="--thrust-min" && need(i)) a.thrust_min = std::stod(argv[++i]);
        else if (k=="--thrust-max" && need(i)) a.thrust_max = std::stod(argv[++i]);
        else if (k=="--impulse-budget" && need(i)) a.impulse_budget = std::stod(argv[++i]);
        else if (k=="--w-alt" && need(i)) a.w_alt = std::stod(argv[++i]);
        else if (k=="--w-ao" && need(i)) a.w_ao = std::stod(argv[++i]);
        else if (k=="--w-roll" && need(i)) a.w_roll = std::stod(argv[++i]);
        else if (k=="--w-effort" && need(i)) a.w_effort = std::stod(argv[++i]);
        else if (k=="--w-dwell-alt" && need(i)) a.w_dwell_alt = std::stod(argv[++i]);
        else if (k=="--w-dwell-att" && need(i)) a.w_dwell_att = std::stod(argv[++i]);
        else if (k=="--alt-band" && need(i)) a.alt_band = std::stod(argv[++i]);
        else if (k=="--aoa-band" && need(i)) a.aoa_band_deg = std::stod(argv[++i]);
        else if (k=="--aos-band" && need(i)) a.aos_band_deg = std::stod(argv[++i]);
        else if (k=="--roll-band" && need(i)) a.roll_band_deg = std::stod(argv[++i]);
        else if (k=="--omega-band" && need(i)) a.omega_band_deg_s = std::stod(argv[++i]);
        else if (k=="--f107a" && need(i)) a.f107a = std::stof(argv[++i]);
        else if (k=="--f107" && need(i)) a.f107 = std::stof(argv[++i]);
        else if (k=="--ap" && need(i)) {
            std::string s=argv[++i]; std::replace(s.begin(), s.end(), ',', ' '); std::istringstream iss(s); for (int j=0;j<7 && iss; ++j){ iss>>a.ap[j]; }
        }
        else if (k=="--config" && need(i)) a.config_path = argv[++i];
        else if (k=="--log-csv" && need(i)) a.log_csv_path = argv[++i];
        else if (k=="--align-to-velocity") a.align_to_velocity = true;
        else if (k=="--debug-csv" && need(i)) a.debug_csv_path = argv[++i];
        else if (k=="--control-dt" && need(i)) a.control_dt = std::stod(argv[++i]);
        else if (k=="--control-substeps" && need(i)) a.control_substeps = std::stoi(argv[++i]);
        else if (k=="--help" || k=="-h") { print_usage(); return false; }
        else { std::cerr<<"Unknown arg: "<<k<<"\n"; print_usage(); return false; }
    }
    return true;
}

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

static void apply_json_config(const std::string& js, Args& a) {
    std::string integ; if (find_string(js, "integrator", integ)) a.integrator = integ;
    double tmp;
    if (find_number(js, "episodes", tmp)) a.episodes = (int)tmp;
    if (find_number(js, "orbits", tmp)) a.orbits = (int)tmp;
    if (find_number(js, "seed", tmp)) a.seed = (uint64_t)tmp;
    if (find_number(js, "dt", tmp)) a.dt = tmp;
    if (find_number(js, "target_altitude", tmp)) a.target_alt = tmp;
    if (find_number(js, "eta_limit_deg", tmp)) a.eta_limit_deg = tmp;
    if (find_number(js, "thrust_min", tmp)) a.thrust_min = tmp;
    if (find_number(js, "thrust_max", tmp)) a.thrust_max = tmp;
    if (find_number(js, "impulse_budget", tmp)) a.impulse_budget = tmp;
    if (find_number(js, "w_alt", tmp)) a.w_alt = tmp;
    if (find_number(js, "w_ao", tmp)) a.w_ao = tmp;
    if (find_number(js, "w_roll", tmp)) a.w_roll = tmp;
    if (find_number(js, "w_effort", tmp)) a.w_effort = tmp;
    if (find_number(js, "w_dwell_alt", tmp)) a.w_dwell_alt = tmp;
    if (find_number(js, "w_dwell_att", tmp)) a.w_dwell_att = tmp;
    if (find_number(js, "alt_band", tmp)) a.alt_band = tmp;
    if (find_number(js, "aoa_band_deg", tmp)) a.aoa_band_deg = tmp;
    if (find_number(js, "aos_band_deg", tmp)) a.aos_band_deg = tmp;
    if (find_number(js, "roll_band_deg", tmp)) a.roll_band_deg = tmp;
    if (find_number(js, "omega_band_deg_s", tmp)) a.omega_band_deg_s = tmp;
    if (find_number(js, "alt_min", tmp)) a.alt_min = tmp;
    if (find_number(js, "alt_max", tmp)) a.alt_max = tmp;
    if (find_number(js, "incl_min_deg", tmp)) a.incl_min_deg = tmp;
    if (find_number(js, "incl_max_deg", tmp)) a.incl_max_deg = tmp;
    double ftmp;
    if (find_number(js, "f107a", tmp)) a.f107a = (float)tmp;
    if (find_number(js, "f107", tmp)) a.f107 = (float)tmp;
    find_array7(js, "ap", a.ap);
}

int main(int argc, char** argv) {
    Args args; if (!parse_args(argc, argv, args)) return 1;
    if (!args.config_path.empty()) {
        std::string js = slurp(args.config_path);
        if (js.empty()) { std::cerr<<"Failed to read config: "<<args.config_path<<"\n"; return 1; }
        apply_json_config(js, args);
    }

    EnvConfig cfg;
    cfg.init_altitude_min_m = args.alt_min;
    cfg.init_altitude_max_m = args.alt_max;
    cfg.init_incl_min_rad = deg2rad(args.incl_min_deg);
    cfg.init_incl_max_rad = deg2rad(args.incl_max_deg);
    cfg.target_altitude_m = args.target_alt;
    cfg.eta_limit_rad = deg2rad(args.eta_limit_deg);
    cfg.thrust_min_N = args.thrust_min;
    cfg.thrust_max_N = args.thrust_max;
    cfg.total_impulse_budget_Ns = args.impulse_budget;
    cfg.N_orbits_per_episode = args.orbits;
    cfg.dt_initial_s = args.dt;
    cfg.space_weather.f107a = args.f107a;
    cfg.space_weather.f107 = args.f107;
    cfg.space_weather.ap = args.ap;
    cfg.w_alt = args.w_alt; cfg.w_att_ao = args.w_ao; cfg.w_att_roll = args.w_roll; cfg.w_effort = args.w_effort; cfg.w_dwell_alt = args.w_dwell_alt; cfg.w_dwell_att = args.w_dwell_att;
    cfg.alt_band_halfwidth_m = args.alt_band;
    cfg.aoa_band_rad = deg2rad(args.aoa_band_deg);
    cfg.aos_band_rad = deg2rad(args.aos_band_deg);
    cfg.roll_band_rad = deg2rad(args.roll_band_deg);
    cfg.omega_band_rad_s = deg2rad(args.omega_band_deg_s);
    cfg.align_attitude_to_velocity = args.align_to_velocity;

    PropEnv env; env.init(cfg);
    if (!args.debug_csv_path.empty()) env.set_debug_csv(args.debug_csv_path);
    if (args.integrator == "rk4") env.set_integrator(IntegratorType::RK4);
    else env.set_integrator(IntegratorType::DP54, DP54Params{});

    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> unif_eta(-cfg.eta_limit_rad, cfg.eta_limit_rad);
    std::uniform_real_distribution<double> unif_thrust(cfg.thrust_min_N, cfg.thrust_max_N);

    std::ofstream csv;
    if (!args.log_csv_path.empty()) {
        csv.open(args.log_csv_path);
        csv << "episode,orbit,integrator,dt,eta1,eta2,thrust,rem_impulse,substeps,"
               "R_total,R_alt,R_att,R_spin,R_effort,dwell_alt_frac,dwell_att_frac,"
               "altitude,AoA,AoS,roll\n";
    }

    for (int ep=0; ep<args.episodes; ++ep) {
        uint64_t seed = args.seed + ep;
        double jd0 = 2451545.0 + (seed % 86400) / 86400.0; // vary UT
        Observation obs0 = env.reset_random(seed, jd0);
        std::cout << "Episode "<<ep
                  << ": alt0="<<obs0.altitude_m<<" m"
                  << ", AoA0="<<obs0.aoa_rad<<" rad"
                  << ", AoS0="<<obs0.aos_rad<<" rad"
                  << ", roll0="<<obs0.roll_rad<<" rad"
                  << ", target_alt="<<cfg.target_altitude_m
                  << ", bands: alt±"<<cfg.alt_band_halfwidth_m<<" m, AoA±"<<args.aoa_band_deg
                  << " deg, AoS±"<<args.aos_band_deg<<" deg, roll±"<<args.roll_band_deg
                  << " deg, |ω|≤"<<args.omega_band_deg_s<<" deg/s"
                  << "\n";
        bool done=false;
        if (args.control_substeps > 0) {
            double T_est = env.estimate_period_s();
            double total_time = T_est * cfg.N_orbits_per_episode;
            double t_elapsed = 0.0;
            int step_idx = 0;
            while (t_elapsed < total_time && !done) {
                Controls u; u.eta1_rad = unif_eta(rng); u.eta2_rad = unif_eta(rng); u.thrust_N = (cfg.thrust_max_N>0? unif_thrust(rng) : 0.0);
                StepResult sr = env.step_substeps(u, args.control_substeps);
                std::cout << "  Control step "<<step_idx
                          << ": action(eta1="<<u.eta1_rad<<", eta2="<<u.eta2_rad<<", thrust="<<u.thrust_N<<" N)"
                          << ", substeps="<<sr.substeps
                          << ", reward total="<<sr.rew.total
                          << " [alt="<<sr.rew.r_alt<<", att="<<sr.rew.r_att<<", spin="<<sr.rew.r_spin<<", effort="<<sr.rew.r_effort<<", dwell="<< (sr.rew.total - (sr.rew.r_alt+sr.rew.r_att+sr.rew.r_spin+sr.rew.r_effort)) << "]"
                          << ", dwell_alt="<<sr.dwell_alt_frac*100.0<<"%"
                          << ", dwell_att="<<sr.dwell_att_frac*100.0<<"%"
                          << ", rem_impulse="<<sr.remaining_impulse_Ns<<" Ns"
                          << ", alt="<<sr.obs.altitude_m
                          << ", AoA="<<sr.obs.aoa_rad<<", AoS="<<sr.obs.aos_rad<<", roll="<<sr.obs.roll_rad
                          << (sr.done? " [TERMINATED]" : "")
                          << "\n";
                if (csv.is_open()) {
                    csv << ep << ',' << step_idx << ',' << args.integrator << ',' << args.dt << ','
                        << u.eta1_rad << ',' << u.eta2_rad << ',' << u.thrust_N << ',' << sr.remaining_impulse_Ns << ',' << sr.substeps << ','
                        << sr.rew.total << ',' << sr.rew.r_alt << ',' << sr.rew.r_att << ',' << sr.rew.r_spin << ',' << sr.rew.r_effort << ','
                        << sr.dwell_alt_frac << ',' << sr.dwell_att_frac << ','
                        << sr.obs.altitude_m << ',' << sr.obs.aoa_rad << ',' << sr.obs.aos_rad << ',' << sr.obs.roll_rad
                        << '\n';
                }
                done = sr.done;
                t_elapsed += env.last_T_est_;
                ++step_idx;
            }
        } else if (args.control_dt > 0.0) {
            double T_est = env.estimate_period_s();
            double total_time = T_est * cfg.N_orbits_per_episode;
            double t_elapsed = 0.0;
            int step_idx = 0;
            while (t_elapsed < total_time && !done) {
                Controls u; u.eta1_rad = unif_eta(rng); u.eta2_rad = unif_eta(rng); u.thrust_N = (cfg.thrust_max_N>0? unif_thrust(rng) : 0.0);
                StepResult sr = env.step_duration(u, std::min(args.control_dt, total_time - t_elapsed));
                std::cout << "  Control step "<<step_idx
                          << ": action(eta1="<<u.eta1_rad<<", eta2="<<u.eta2_rad<<", thrust="<<u.thrust_N<<" N)"
                          << ", substeps="<<sr.substeps
                          << ", reward total="<<sr.rew.total
                          << " [alt="<<sr.rew.r_alt<<", att="<<sr.rew.r_att<<", spin="<<sr.rew.r_spin<<", effort="<<sr.rew.r_effort<<", dwell="<< (sr.rew.total - (sr.rew.r_alt+sr.rew.r_att+sr.rew.r_spin+sr.rew.r_effort)) << "]"
                          << ", dwell_alt="<<sr.dwell_alt_frac*100.0<<"%"
                          << ", dwell_att="<<sr.dwell_att_frac*100.0<<"%"
                          << ", rem_impulse="<<sr.remaining_impulse_Ns<<" Ns"
                          << ", alt="<<sr.obs.altitude_m
                          << ", AoA="<<sr.obs.aoa_rad<<", AoS="<<sr.obs.aos_rad<<", roll="<<sr.obs.roll_rad
                          << (sr.done? " [TERMINATED]" : "")
                          << "\n";
                if (csv.is_open()) {
                    csv << ep << ',' << step_idx << ',' << args.integrator << ',' << args.dt << ','
                        << u.eta1_rad << ',' << u.eta2_rad << ',' << u.thrust_N << ',' << sr.remaining_impulse_Ns << ',' << sr.substeps << ','
                        << sr.rew.total << ',' << sr.rew.r_alt << ',' << sr.rew.r_att << ',' << sr.rew.r_spin << ',' << sr.rew.r_effort << ','
                        << sr.dwell_alt_frac << ',' << sr.dwell_att_frac << ','
                        << sr.obs.altitude_m << ',' << sr.obs.aoa_rad << ',' << sr.obs.aos_rad << ',' << sr.obs.roll_rad
                        << '\n';
                }
                done = sr.done; t_elapsed += env.last_T_est_; ++step_idx;
            }
        } else {
            for (int k=0; k<cfg.N_orbits_per_episode && !done; ++k) {
                Controls u; u.eta1_rad = unif_eta(rng); u.eta2_rad = unif_eta(rng); u.thrust_N = (cfg.thrust_max_N>0? unif_thrust(rng) : 0.0);
                StepResult sr = env.step_orbit(u);
                std::cout << "  Orbit "<<k
                          << ": action(eta1="<<u.eta1_rad<<", eta2="<<u.eta2_rad<<", thrust="<<u.thrust_N<<" N)"
                          << ", substeps="<<sr.substeps
                          << ", reward total="<<sr.rew.total
                          << " [alt="<<sr.rew.r_alt<<", att="<<sr.rew.r_att<<", spin="<<sr.rew.r_spin<<", effort="<<sr.rew.r_effort<<", dwell="<< (sr.rew.total - (sr.rew.r_alt+sr.rew.r_att+sr.rew.r_spin+sr.rew.r_effort)) << "]"
                          << ", dwell_alt="<<sr.dwell_alt_frac*100.0<<"%"
                          << ", dwell_att="<<sr.dwell_att_frac*100.0<<"%"
                          << ", rem_impulse="<<sr.remaining_impulse_Ns<<" Ns"
                          << ", alt="<<sr.obs.altitude_m
                          << ", AoA="<<sr.obs.aoa_rad<<", AoS="<<sr.obs.aos_rad<<", roll="<<sr.obs.roll_rad
                          << (sr.done? " [TERMINATED]" : "")
                          << "\n";
                if (csv.is_open()) {
                    csv << ep << ',' << k << ',' << args.integrator << ',' << args.dt << ','
                        << u.eta1_rad << ',' << u.eta2_rad << ',' << u.thrust_N << ',' << sr.remaining_impulse_Ns << ',' << sr.substeps << ','
                        << sr.rew.total << ',' << sr.rew.r_alt << ',' << sr.rew.r_att << ',' << sr.rew.r_spin << ',' << sr.rew.r_effort << ','
                        << sr.dwell_alt_frac << ',' << sr.dwell_att_frac << ','
                        << sr.obs.altitude_m << ',' << sr.obs.aoa_rad << ',' << sr.obs.aos_rad << ',' << sr.obs.roll_rad
                        << '\n';
                }
                done = sr.done;
            }
        }
    }
    return 0;
}
