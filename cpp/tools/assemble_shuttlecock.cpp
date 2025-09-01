#include "show_bodies.h"
#include "body_importer.h"
#include <Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>

using namespace vleo_aerodynamics_core;

static std::string resolve_path_ancestor(const std::string& name) {
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
    double eta1_deg = 0.0, eta2_deg = 0.0;
    std::string out = "assembled.obj";
    if (argc >= 2) eta1_deg = std::stod(argv[1]);
    if (argc >= 3) eta2_deg = std::stod(argv[2]);
    if (argc >= 4) out = argv[3];

    std::vector<std::string> object_files = {
        resolve_path_ancestor("shuttle_box.obj"),
        resolve_path_ancestor("wing_pos_x.obj"),
        resolve_path_ancestor("wing_neg_x.obj"),
        resolve_path_ancestor("wing_pos_y.obj"),
        resolve_path_ancestor("wing_neg_y.obj"),
    };

    const double box_half = 0.05;
    Eigen::Matrix3Xd hinge(3, 5), axes(3, 5);
    hinge.col(0) << 0.0, 0.0, 0.0;
    hinge.col(1) << +box_half, 0.0, 0.0;
    hinge.col(2) << -box_half, 0.0, 0.0;
    hinge.col(3) << 0.0, +box_half, 0.0;
    hinge.col(4) << 0.0, -box_half, 0.0;
    axes.setZero();
    axes.col(0) = Eigen::Vector3d::UnitZ();      // box (unused)
    axes.col(1) = Eigen::Vector3d::UnitY();      // +X wing about Y
    axes.col(2) = Eigen::Vector3d::UnitY();      // -X wing about Y
    axes.col(3) = Eigen::Vector3d::UnitX();      // +Y wing about X
    axes.col(4) = Eigen::Vector3d::UnitX();      // -Y wing about X

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
    // signs chosen to rotate outward about per-edge hinge axes
    std::vector<double> angles = {0.0, -eta1, +eta1, +eta2, -eta2};

    try {
        showBodies(bodies, angles, out);
    } catch (const std::exception& e) {
        std::cerr << "Export error: " << e.what() << std::endl;
        return 2;
    }
    std::cout << "Wrote " << out << std::endl;
    return 0;
}
