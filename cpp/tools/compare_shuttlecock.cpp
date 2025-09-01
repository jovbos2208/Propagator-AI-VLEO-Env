#include "vleo_aerodynamics.h"
#include "show_bodies.h"
#include "body_importer.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace vleo_aerodynamics_core;

static std::string resolve_ancestor(const std::string& name) {
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
    return name;
}

int main(int argc, char** argv) {
    // Defaults: both wing pairs at 90 deg, print CSV, optional compare-file
    double eta1_deg = 90.0, eta2_deg = 90.0; // X-wings, Y-wings
    bool print_csv = true;
    std::string compare_file;
    double atol = 1e-6, rtol = 1e-5;
    std::string export_obj;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](double& tgt) {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return false; }
            tgt = std::atof(argv[++i]);
            return true;
        };
        if (a == "--eta1-deg") { if (!need_val(eta1_deg)) return 2; }
        else if (a == "--eta2-deg") { if (!need_val(eta2_deg)) return 2; }
        else if (a == "--print-csv") { print_csv = true; }
        else if (a == "--no-csv") { print_csv = false; }
        else if (a == "--compare-file") { if (i + 1 >= argc) { std::cerr << "Missing value for --compare-file\n"; return 2; } compare_file = argv[++i]; }
        else if (a == "--atol") { if (!need_val(atol)) return 2; }
        else if (a == "--rtol") { if (!need_val(rtol)) return 2; }
        else if (a == "--export-obj") { if (i + 1 >= argc) { std::cerr << "Missing value for --export-obj\n"; return 2; } export_obj = argv[++i]; }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: compare_shuttlecock [--eta1-deg d] [--eta2-deg d] [--compare-file path] [--atol v] [--rtol v] [--export-obj path] [--print-csv|--no-csv]\n";
            return 0;
        }
    }

    std::vector<std::string> object_files = {
        resolve_ancestor("shuttle_box.obj"),
        resolve_ancestor("wing_pos_x.obj"),
        resolve_ancestor("wing_neg_x.obj"),
        resolve_ancestor("wing_pos_y.obj"),
        resolve_ancestor("wing_neg_y.obj"),
    };

    const double box_half = 0.05;
    Eigen::Matrix3Xd hinge(3, 5), axes(3, 5);
    hinge.col(0) << 0.0, 0.0, 0.0;
    hinge.col(1) << +box_half, 0.0, 0.0;
    hinge.col(2) << -box_half, 0.0, 0.0;
    hinge.col(3) << 0.0, +box_half, 0.0;
    hinge.col(4) << 0.0, -box_half, 0.0;
    axes.setZero();
    axes.col(0) = Eigen::Vector3d::UnitZ();
    axes.col(1) = Eigen::Vector3d::UnitY();
    axes.col(2) = Eigen::Vector3d::UnitY();
    axes.col(3) = Eigen::Vector3d::UnitX();
    axes.col(4) = Eigen::Vector3d::UnitX();

    std::vector<std::vector<double>> temps_K(5, std::vector<double>{300.0});
    std::vector<std::vector<double>> eac(5, std::vector<double>{0.9});
    Eigen::Matrix3d DCM = Eigen::Matrix3d::Identity();
    Eigen::Vector3d CoM = Eigen::Vector3d::Zero();

    std::vector<Body> bodies;
    try {
        bodies = importMultipleBodies(object_files, hinge, axes, temps_K, eac, DCM, CoM);
    } catch (const std::exception& e) {
        std::cerr << "Import error: " << e.what() << std::endl;
        return 1;
    }

    double eta1 = eta1_deg * M_PI / 180.0;
    double eta2 = eta2_deg * M_PI / 180.0;
    std::vector<double> angles = {0.0, -eta1, +eta1, +eta2, -eta2};

    if (!export_obj.empty()) {
        try { showBodies(bodies, angles, export_obj); } catch (...) {}
    }

    // Aerodynamics inputs consistent with MATLAB
    Eigen::Quaterniond q(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d omega(0.0, 0.0, 0.0);
    Eigen::Vector3d V(7500.0, 0.0, 0.0);
    Eigen::Vector3d wind(0.0, 0.0, 0.0);
    double rho = 1e-12;
    double T = 1000.0;
    double m = 16.0 * 1.66053906660e-27;
    int Tmethod = 2;

    auto res = vleoAerodynamics(q, omega, V, wind, rho, T, m, bodies, angles, Tmethod);

    if (print_csv) {
        std::cout << "force," << res.force.transpose() << "\n";
        std::cout << "torque," << res.torque.transpose() << "\n";
    }

    if (!compare_file.empty()) {
        std::ifstream in(compare_file);
        if (!in) { std::cerr << "Failed to open compare file: " << compare_file << std::endl; return 3; }
        Eigen::Vector3d Fexp, Texp;
        std::string line;
        auto parse_vec = [](const std::string& s, Eigen::Vector3d& out){
            std::string t = s; std::replace(t.begin(), t.end(), ',', ' ');
            std::istringstream iss(t); double x,y,z; if (!(iss>>x>>y>>z)) return false; out=Eigen::Vector3d(x,y,z); return true; };
        bool gotF=false, gotT=false;
        while (std::getline(in, line)) {
            if (line.rfind("force", 0) == 0) {
                auto pos=line.find_first_of(",;\t "); if (pos==std::string::npos) continue; gotF=parse_vec(line.substr(pos+1), Fexp);
            } else if (line.rfind("torque", 0) == 0) {
                auto pos=line.find_first_of(",;\t "); if (pos==std::string::npos) continue; gotT=parse_vec(line.substr(pos+1), Texp);
            }
        }
        if (!gotF || !gotT) { std::cerr << "Invalid compare file format." << std::endl; return 4; }
        auto within=[&](double v,double e){ double d=std::abs(v-e); double tol=atol+rtol*std::abs(e); return d<=tol; };
        bool ok=true; for(int i=0;i<3;++i) ok &= within(res.force(i),Fexp(i)); for(int i=0;i<3;++i) ok &= within(res.torque(i),Texp(i));
        if (!ok) {
            std::cerr << "Comparison failed (atol="<<atol<<", rtol="<<rtol<<")\n";
            std::cerr << "F got: "<<res.force.transpose()<<" exp: "<<Fexp.transpose()<<"\n";
            std::cerr << "T got: "<<res.torque.transpose()<<" exp: "<<Texp.transpose()<<"\n";
            return 5;
        }
        std::cout << "Comparison passed." << std::endl;
    }

    return 0;
}

