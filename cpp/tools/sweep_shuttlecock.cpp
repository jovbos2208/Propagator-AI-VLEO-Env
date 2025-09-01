#include "vleo_aerodynamics.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace vleo_aerodynamics_core;

static std::string resolve_example_path(const std::string& name) {
    namespace fs = std::filesystem;
    fs::path base = fs::current_path();
    const std::vector<fs::path> rels = {
        fs::path("vleo-aerodynamics-c-export/vleo-aerodynamics-tool/example/obj_files") / name,
        fs::path("vleo-aerodynamics-c-export/vleo-aerodynamics-tool/example/gmsh_files") / name,
        fs::path("cpp/tests") / name,
        fs::path("tests") / name,
    };
    for (int up = 0; up < 6; ++up) {
        for (const auto& rel : rels) {
            fs::path cand = base / rel;
            if (fs::exists(cand)) return cand.string();
        }
        if (!base.has_parent_path()) break;
        base = base.parent_path();
    }
    return name;
}

int main(int argc, char** argv) {
    // Geometry file order must match hinge/axis columns: [MainBody, WingRight, WingTop, WingLeft, WingBottom]
    std::vector<std::string> object_files = {
        resolve_example_path("MainBody.obj"),
        resolve_example_path("WingRight.obj"),
        resolve_example_path("WingTop.obj"),
        resolve_example_path("WingLeft.obj"),
        resolve_example_path("WingBottom.obj"),
    };

    // Rotation hinge points and directions in CAD frame (from MATLAB example)
    Eigen::Matrix3Xd hinge(3, 5), axes(3, 5);
    hinge << 0.00,  0.05, 0.00, -0.05,  0.00,
             0.00,  0.00, 0.05,  0.00, -0.05,
             0.00,  0.00, 0.00,  0.00,  0.00;
    axes  << 1, 0, 1, 0, -1,
             0, 1, 0, 1,  0,
             0, 0, 0, 0,  0;

    // Temperatures and energy accommodation
    std::vector<std::vector<double>> temps_K(5, std::vector<double>{300.0});
    std::vector<std::vector<double>> eac(5, std::vector<double>{0.9});

    // Orientation and origin (same as example.m)
    Eigen::Matrix3d DCM;
    DCM << 0, -1, 0,
           -1, 0, 0,
            0,  0, -1;
    Eigen::Vector3d CoM = Eigen::Vector3d::Zero();

    // Import bodies
    std::vector<Body> bodies;
    try {
        bodies = importMultipleBodies(object_files, hinge, axes, temps_K, eac, DCM, CoM);
    } catch (const std::exception& e) {
        std::cerr << "Import error: " << e.what() << std::endl;
        return 1;
    }

    // Environment and constants
    const double altitude_m = 3e5;
    const double mu = 3.986e14;           // m^3/s^2
    const double earth_radius_m = 6.378e6;
    const double V_orbit = std::sqrt(mu / (earth_radius_m + altitude_m));
    const Eigen::Vector3d V_I(V_orbit, 0.0, 0.0);
    const Eigen::Vector3d wind_I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d omega_BI_B = Eigen::Vector3d::Zero();

    // Gas params
    double rho = 1e-12;             // kg/m^3
    double T_K = 1000.0;            // K
    double particles_mass = 16.0 * 1.6605390689252e-27; // kg
    int temp_ratio_method = 1;

    // Optional CLI overrides
    auto need_value = [&](int i){ if (i+1 >= argc) { std::cerr << "Missing value for " << argv[i] << "\n"; std::exit(1);} };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rho") { need_value(i); rho = std::stod(argv[++i]); }
        else if (arg == "--T" || arg == "--T_K") { need_value(i); T_K = std::stod(argv[++i]); }
        else if (arg == "--mass") { need_value(i); particles_mass = std::stod(argv[++i]); }
        else if (arg == "--method") { need_value(i); temp_ratio_method = std::stoi(argv[++i]); }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sweep_shuttlecock [--rho <kg/m^3>] [--T <K>] [--mass <kg>] [--method {1|2|3}]\n";
            return 0;
        }
    }

    // Grid sweep over wing angles, AoA, and T_K
    const int n_eta = 101;     // 0..100 deg inclusive
    const int n_aoa = 31;      // -15..15 deg inclusive
    const int n_T   = 13;      // 600..1200 K inclusive
    const double eta_min_deg = 0.0,   eta_max_deg = 100.0;
    const double aoa_min_deg = -15.0, aoa_max_deg = 15.0;
    const double T_min = 600.0, T_max = 1200.0;

    namespace fs = std::filesystem;
    fs::create_directories("cpp/build");
    std::ofstream csv_grid("cpp/build/sweep_results.csv");
    csv_grid << "eta1_deg,eta2_deg,aoa_deg,T_K,Fx,Fy,Fz,Tx,Ty,Tz\n";

    const Eigen::Vector3d y_axis = Eigen::Vector3d::UnitY();

    const long long total = static_cast<long long>(n_eta) * n_eta * n_aoa * n_T;
    long long done = 0;
    const int barWidth = 50;
    auto show_progress = [&](long long idx){
        double ratio = (total <= 1) ? 1.0 : (static_cast<double>(idx) / total);
        int filled = static_cast<int>(ratio * barWidth);
        std::cerr << '\r' << "Grid sweep "
                  << '[' << std::string(filled, '=') << std::string(barWidth - filled, ' ') << "] "
                  << std::setw(3) << static_cast<int>(ratio * 100.0) << "% (" << idx << ":" << total << ")" << std::flush;
    };

    for (int iT = 0; iT < n_T; ++iT) {
        double tT = (n_T == 1) ? 0.0 : (static_cast<double>(iT) / (n_T - 1));
        double T_now = T_min + tT * (T_max - T_min);
        for (int ia = 0; ia < n_aoa; ++ia) {
            double ta = (n_aoa == 1) ? 0.0 : (static_cast<double>(ia) / (n_aoa - 1));
            double aoa_deg = aoa_min_deg + ta * (aoa_max_deg - aoa_min_deg);
            double aoa_rad = aoa_deg * M_PI / 180.0;
            Eigen::Quaterniond q_BI(std::cos(aoa_rad / 2.0),
                                    y_axis.x() * std::sin(aoa_rad / 2.0),
                                    y_axis.y() * std::sin(aoa_rad / 2.0),
                                    y_axis.z() * std::sin(aoa_rad / 2.0));
            for (int i1 = 0; i1 < n_eta; ++i1) {
                double t1 = (n_eta == 1) ? 0.0 : (static_cast<double>(i1) / (n_eta - 1));
                double eta1_deg = eta_min_deg + t1 * (eta_max_deg - eta_min_deg);
                double eta1 = eta1_deg * M_PI / 180.0;
                for (int i2 = 0; i2 < n_eta; ++i2) {
                    double t2 = (n_eta == 1) ? 0.0 : (static_cast<double>(i2) / (n_eta - 1));
                    double eta2_deg = eta_min_deg + t2 * (eta_max_deg - eta_min_deg);
                    double eta2 = eta2_deg * M_PI / 180.0;

                    std::vector<double> angles(5, 0.0);
                    // Opposite wings share same absolute value (signs chosen for symmetric deployment)
                    angles[1] = -eta1; // WingRight
                    angles[3] = +eta1; // WingLeft
                    angles[2] = +eta2; // WingTop
                    angles[4] = -eta2; // WingBottom

                    AeroForceAndTorque out;
                    try {
                        out = vleoAerodynamics(q_BI, omega_BI_B, V_I, wind_I,
                                               rho, T_now, particles_mass,
                                               bodies, angles, temp_ratio_method);
                    } catch (const std::exception& e) {
                        std::cerr << "Eval error at (T_i="<<iT<<", AoA_i="<<ia<<", eta1_i="<<i1<<", eta2_i="<<i2<<"): "
                                  << e.what() << std::endl;
                        ++done; if ((done % 1000) == 0 || done == total) show_progress(done);
                        continue;
                    }

                    csv_grid << eta1_deg << ',' << eta2_deg << ',' << aoa_deg << ',' << T_now << ','
                             << out.force.x() << ',' << out.force.y() << ',' << out.force.z() << ','
                             << out.torque.x() << ',' << out.torque.y() << ',' << out.torque.z() << '\n';

                    ++done;
                    if ((done % 1000) == 0 || done == total) show_progress(done);
                }
            }
        }
    }

    std::cerr << '\n';
    std::cout << "Wrote cpp/build/sweep_results.csv" << std::endl;
    return 0;
}
