#include "shuttlecock_aero.h"
#include "vleo_aerodynamics.h"
#include <vector>
#include <string>
#include <cstdio>
#include <filesystem>

namespace vleo_aerodynamics_core {

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

std::pair<Eigen::Vector3d, Eigen::Vector3d> shuttlecock_aero(double Vx, double Vy, double Vz,
                                                             double eta1_rad, double eta2_rad,
                                                             double rho, double T_K, double s) {
    // Geometry files (box + 4 wings)
    std::vector<std::string> object_files = {
        resolve_path_ancestor("shuttle_box.obj"),
        resolve_path_ancestor("wing_pos_x.obj"),
        resolve_path_ancestor("wing_neg_x.obj"),
        resolve_path_ancestor("wing_pos_y.obj"),
        resolve_path_ancestor("wing_neg_y.obj"),
    };

    // Box dimensions (match OBJ): half-length = 0.05
    const double box_half = 0.05;

    // Hinge points and rotation axes (about Z, outward from box edges)
    Eigen::Matrix3Xd hinge(3, 5), axes(3, 5);
    hinge.col(0) << 0.0, 0.0, 0.0;     // box unused
    hinge.col(1) << +box_half, 0.0, 0.0; // +X wing hinge line through (1,0,0)
    hinge.col(2) << -box_half, 0.0, 0.0; // -X wing hinge line through (-1,0,0)
    hinge.col(3) << 0.0, +box_half, 0.0; // +Y wing hinge line
    hinge.col(4) << 0.0, -box_half, 0.0; // -Y wing hinge line
    axes.setZero();
    // Axes per-wing along the actual box edge line:
    // +X / -X wings: hinge edge runs along +Y (vertical in 2D), rotate about +Y
    axes.col(0) = Eigen::Vector3d::UnitZ();      // box (unused)
    axes.col(1) = Eigen::Vector3d::UnitY();      // +X wing about Y
    axes.col(2) = Eigen::Vector3d::UnitY();      // -X wing about Y
    // +Y / -Y wings: hinge edge runs along +X (horizontal in 2D), rotate about +X
    axes.col(3) = Eigen::Vector3d::UnitX();      // +Y wing about X
    axes.col(4) = Eigen::Vector3d::UnitX();      // -Y wing about X

    // Temperatures and energy accommodation
    std::vector<std::vector<double>> temps_K(5, std::vector<double>{300.0});
    std::vector<std::vector<double>> eac(5, std::vector<double>{0.9});

    // Identity DCM and CoM
    Eigen::Matrix3d DCM = Eigen::Matrix3d::Identity();
    Eigen::Vector3d CoM = Eigen::Vector3d::Zero();

    // Import all bodies
    auto bodies = importMultipleBodies(object_files, hinge, axes, temps_K, eac, DCM, CoM);

    // Apply rotation angles: choose signs so wings rotate outward about z-axis hinges
    // Order: [box, +X wing, -X wing, +Y wing, -Y wing]
    std::vector<double> angles = {0.0, +eta1_rad, -eta1_rad, +eta2_rad, -eta2_rad};

    // Velocity and wind in I-frame
    Eigen::Vector3d V_I(Vx, Vy, Vz);
    Eigen::Vector3d wind_I = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_BI(1.0, 0.0, 0.0, 0.0); // body aligned with inertial
    Eigen::Vector3d omega_BI_B = Eigen::Vector3d::Zero();

    // Infer particle mass from s = |V| / cm, cm = sqrt(2 kB T / m)
    const double kB = 1.38064852e-23;
    double V = V_I.norm();
    if (V <= 0.0 || s <= 0.0) {
        return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    }
    // m = 2 kB T (s / V)^2
    double m_particle = 2.0 * kB * T_K * (s / V) * (s / V);

    // Use temperature ratio method 2 by default
    int temp_ratio_method = 2;

    auto out = vleoAerodynamics(q_BI, omega_BI_B, V_I, wind_I, rho, T_K, m_particle,
                                 bodies, angles, temp_ratio_method);
    return {out.force, out.torque};
}

}
