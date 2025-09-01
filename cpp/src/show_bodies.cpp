#include "show_bodies.h"
#include "rotate_body.h"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace vleo_aerodynamics_core {

void showBodies(const std::vector<Body>& bodies,
                const std::vector<double>& bodies_rotation_angles_rad,
                const std::string& out_obj_path) {
    // Ensure parent directory exists
    namespace fs = std::filesystem;
    fs::path outp(out_obj_path);
    if (!outp.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(outp.parent_path(), ec);
    }

    std::ofstream out(out_obj_path);
    if (!out) {
        throw std::runtime_error("Failed to open output OBJ file: " + out_obj_path);
    }

    // Write a simple OBJ with separate objects per body
    out << "# Rotated bodies export\n";

    size_t vertex_offset = 0; // OBJ indices are 1-based

    for (size_t i = 0; i < bodies.size(); ++i) {
        // Copy and rotate current body data
        Eigen::Matrix3Xd vertices = bodies[i].vertices_B;
        std::vector<Eigen::Vector3d> centroids = bodies[i].centroids_B;
        std::vector<Eigen::Vector3d> normals = bodies[i].normals_B;

        rotateBody(vertices, centroids, normals,
                   bodies_rotation_angles_rad[i],
                   bodies[i].rotation_direction_B,
                   bodies[i].rotation_hinge_point_B);

        out << "o body_" << i << "\n";

        // Write vertices: 3 per triangle
        int num_cols = static_cast<int>(vertices.cols());
        for (int c = 0; c < num_cols; ++c) {
            out << "v "
                << vertices(0, c) << ' '
                << vertices(1, c) << ' '
                << vertices(2, c) << "\n";
        }

        // Write faces: each 3 consecutive columns is one triangle
        int num_tris = num_cols / 3;
        for (int t = 0; t < num_tris; ++t) {
            size_t a = vertex_offset + (3 * t + 1);
            size_t b = vertex_offset + (3 * t + 2);
            size_t c = vertex_offset + (3 * t + 3);
            out << "f " << a << ' ' << b << ' ' << c << "\n";
        }

        vertex_offset += static_cast<size_t>(num_cols);
    }

    out.close();
}

}
