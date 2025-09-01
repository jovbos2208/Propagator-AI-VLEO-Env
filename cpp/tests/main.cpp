#include "vleo_aerodynamics.h"
#include "show_bodies.h"
#include "shuttlecock_aero.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

int main(int argc, char** argv) {
    using namespace vleo_aerodynamics_core;

    // 1) Import body geometry: box with two wings, each as own OBJ
    std::cout << "[1/3] Importing bodies..." << std::endl;
    // Resolve OBJ paths robustly regardless of CWD
    auto resolve_ancestor = [](const std::string& name) {
        namespace fs = std::filesystem;
        fs::path base = fs::current_path();
        for (int up = 0; up < 6; ++up) {
            for (const auto& rel : {fs::path("cpp/tests") / name, fs::path("tests") / name}) {
                fs::path cand = base / rel;
                if (fs::exists(cand)) return cand.string();
            }
            if (!base.has_parent_path()) break;
            base = base.parent_path();
        }
        return name; // fallback; tinyobj will report failure with this name
    };
    std::vector<std::string> object_files = {
        resolve_ancestor("box.obj"),
        resolve_ancestor("wing_left.obj"),
        resolve_ancestor("wing_right.obj"),
    };
    Eigen::Matrix3Xd rotation_hinge_points_CAD(3, 3);
    rotation_hinge_points_CAD.col(0) << 0.0, 0.0, 0.0;  // box
    rotation_hinge_points_CAD.col(1) << -1.0, 0.0, 0.0; // left wing root at x=-1
    rotation_hinge_points_CAD.col(2) <<  1.0, 0.0, 0.0; // right wing root at x=+1
    Eigen::Matrix3Xd rotation_directions_CAD(3, 3);
    rotation_directions_CAD.col(0) << 0.0, 0.0, 1.0; // box (unused)
    rotation_directions_CAD.col(1) << 0.0, 0.0, 1.0; // wings rotate about z
    rotation_directions_CAD.col(2) << 0.0, 0.0, 1.0;
    std::vector<std::vector<double>> temperatures_K = {{300.0}, {300.0}, {300.0}};
    std::vector<std::vector<double>> energy_accommodation_coefficients = {{0.9}, {0.9}, {0.9}};
    Eigen::Matrix3d DCM_B_from_CAD = Eigen::Matrix3d::Identity();
    Eigen::Vector3d CoM_CAD = Eigen::Vector3d::Zero();

    std::vector<Body> bodies;
    try {
        bodies = importMultipleBodies(
            object_files,
            rotation_hinge_points_CAD,
            rotation_directions_CAD,
            temperatures_K,
            energy_accommodation_coefficients,
            DCM_B_from_CAD,
            CoM_CAD);
    } catch (const std::exception& e) {
        std::cerr << "Error during import: " << e.what() << std::endl;
        return 1;
    }
    std::cout << " done.\n";

    // 2) Define inputs to vleoAerodynamics (as in MATLAB test)
    Eigen::Quaterniond attitude_quaternion_BI(1.0, 0.0, 0.0, 0.0); // w, x, y, z
    Eigen::Vector3d rotational_velocity_BI_B(0.0, 0.0, 0.0);
    Eigen::Vector3d velocity_I_I(7500.0, 0.0, 0.0);
    Eigen::Vector3d wind_velocity_I_I(0.0, 0.0, 0.0);
    double density = 1e-12;
    double temperature = 1000.0;
    double particles_mass = 16.0 * 1.66053906660e-27; // Atomic oxygen
    // Parse simple CLI args for rotation angles (degrees) and CSV output
    // Supported flags:
    //   --body-deg <val>
    //   --left-deg <val>
    //   --right-deg <val>
    //   --print-csv
    //   --compare-file <path>  (expects two lines: force,fx,fy,fz and torque,tx,ty,tz)
    //   --atol <val> --rtol <val>
    double body_deg = 0.0, left_deg = 0.0, right_deg = 0.0;
    bool print_csv = false;
    std::string compare_file;
    double atol = 1e-6, rtol = 1e-5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](double& tgt) {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return false; }
            tgt = std::atof(argv[++i]);
            return true;
        };
        if (a == "--body-deg") { if (!need_val(body_deg)) return 3; }
        else if (a == "--left-deg") { if (!need_val(left_deg)) return 3; }
        else if (a == "--right-deg") { if (!need_val(right_deg)) return 3; }
        else if (a == "--wings90") { left_deg = 90.0; right_deg = -90.0; }
        else if (a == "--print-csv") { print_csv = true; }
        else if (a == "--compare-file") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --compare-file\n"; return 3; }
            compare_file = argv[++i];
        }
        else if (a == "--atol") { if (!need_val(atol)) return 3; }
        else if (a == "--rtol") { if (!need_val(rtol)) return 3; }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: tests [--body-deg deg] [--left-deg deg] [--right-deg deg] [--wings90] [--print-csv] [--compare-file path] [--atol v] [--rtol v]\n";
            return 0;
        }
    }
    std::vector<double> bodies_rotation_angles_rad = {
        body_deg * M_PI / 180.0,
        left_deg * M_PI / 180.0,
        right_deg * M_PI / 180.0
    };
    int temperature_ratio_method = 2;

    // 3) Compute forces and torques
    std::cout << "[2/3] Computing aerodynamics..." << std::endl;
    AeroForceAndTorque res;
    try {
        res = vleoAerodynamics(
            attitude_quaternion_BI,
            rotational_velocity_BI_B,
            velocity_I_I,
            wind_velocity_I_I,
            density,
            temperature,
            particles_mass,
            bodies,
            bodies_rotation_angles_rad,
            temperature_ratio_method);
    } catch (const std::exception& e) {
        std::cerr << "Error during aerodynamics: " << e.what() << std::endl;
        return 2;
    }
    std::cout << " done.\n";

    // 4) Print results
    // Optional: export rotated geometry to OBJ to emulate showBodies
    try {
        vleo_aerodynamics_core::showBodies(bodies, bodies_rotation_angles_rad, "scene_rotated.obj");
        std::cout << "[export] Wrote rotated geometry to scene_rotated.obj" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Export error: " << e.what() << std::endl;
    }

    std::cout << "[3/3] Results:" << std::endl;
    if (print_csv) {
        std::cout << "force," << res.force.transpose() << "\n";
        std::cout << "torque," << res.torque.transpose() << "\n";
    } else {
        std::cout << "Aerodynamic Force (N):\n" << res.force.transpose() << std::endl;
        std::cout << "Aerodynamic Torque (Nm):\n" << res.torque.transpose() << std::endl;
    }

    // Optional comparison mode
    if (!compare_file.empty()) {
        auto load_expected = [&](const std::string& path, Eigen::Vector3d& f, Eigen::Vector3d& t) -> bool {
            std::ifstream in(path);
            if (!in) return false;
            std::string line;
            bool gotF=false, gotT=false;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                // normalize separators
                std::replace(line.begin(), line.end(), ';', ',');
                // find prefix
                auto parse_vec = [&](const std::string& payload, Eigen::Vector3d& out) -> bool {
                    std::string s = payload;
                    std::replace(s.begin(), s.end(), ',', ' ');
                    std::istringstream iss(s);
                    double x,y,z;
                    if (!(iss >> x >> y >> z)) return false;
                    out = Eigen::Vector3d(x,y,z);
                    return true;
                };
                if (line.rfind("force", 0) == 0) {
                    auto pos = line.find_first_of(",;\t ");
                    if (pos == std::string::npos) continue;
                    gotF = parse_vec(line.substr(pos+1), f);
                } else if (line.rfind("torque", 0) == 0) {
                    auto pos = line.find_first_of(",;\t ");
                    if (pos == std::string::npos) continue;
                    gotT = parse_vec(line.substr(pos+1), t);
                }
            }
            return gotF && gotT;
        };

        Eigen::Vector3d Fexp, Texp;
        if (!load_expected(compare_file, Fexp, Texp)) {
            std::cerr << "Failed to read expected results from " << compare_file << std::endl;
            return 4;
        }
        auto within = [&](double val, double exp) {
            double diff = std::abs(val - exp);
            double tol = atol + rtol * std::abs(exp);
            return diff <= tol;
        };
        bool ok = true;
        for (int i=0;i<3;++i) ok = ok && within(res.force(i), Fexp(i));
        for (int i=0;i<3;++i) ok = ok && within(res.torque(i), Texp(i));
        if (!ok) {
            std::cerr << "Mismatch vs expected (atol=" << atol << ", rtol=" << rtol << ")\n";
            std::cerr << "F got: " << res.force.transpose() << " exp: " << Fexp.transpose() << "\n";
            std::cerr << "T got: " << res.torque.transpose() << " exp: " << Texp.transpose() << "\n";
            return 5;
        }
        std::cout << "Comparison passed." << std::endl;
    }

    // Also exercise shuttlecock_aero wrapper on the 4-wing setup
    try {
        auto [F, Tq] = vleo_aerodynamics_core::shuttlecock_aero(
            7500.0, 0.0, 0.0,        // Vx,Vy,Vz
            0.3, -0.1,               // eta1, eta2 (x-wings, y-wings)
            1e-12, 1000.0,  // rho, T
            15.0            // s (speed ratio)
        );
        std::cout << "[shuttlecock_aero] Force (N):\n" << F.transpose() << std::endl;
        std::cout << "[shuttlecock_aero] Torque (Nm):\n" << Tq.transpose() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "shuttlecock_aero error: " << e.what() << std::endl;
    }

    return 0;
}
