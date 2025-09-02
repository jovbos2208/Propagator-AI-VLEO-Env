#include "vleo_aerodynamics.h"
#include "smu_stubs.h"
#include "shadowing.h"
#include "rotate_body.h"
#include <iostream>

namespace vleo_aerodynamics_core {

AeroForceAndTorque vleoAerodynamics(const Eigen::Quaterniond& attitude_quaternion_BI,
                                    const Eigen::Vector3d& rotational_velocity_BI_B,
                                    const Eigen::Vector3d& velocity_I_I,
                                    const Eigen::Vector3d& wind_velocity_I_I,
                                    double density,
                                    double temperature,
                                    double particles_mass,
                                    std::vector<Body>& bodies,
                                    const std::vector<double>& bodies_rotation_angles_rad,
                                    int temperature_ratio_method) {

    Eigen::Vector3d v_rel_I = wind_velocity_I_I - velocity_I_I;
    Eigen::Vector3d v_rel_B = smu::transformVector(attitude_quaternion_BI, v_rel_I);

    size_t total_num_faces = 0;
    for (const auto& body : bodies) {
        total_num_faces += body.vertices_B.cols() / 3;
    }

    // Preallocate vertices storage to avoid repeated conservativeResize
    size_t total_vertex_cols = 0;
    for (const auto& body : bodies) {
        total_vertex_cols += static_cast<size_t>(body.vertices_B.cols());
    }
    Eigen::Matrix3Xd all_vertices(3, static_cast<int>(total_vertex_cols));
    std::vector<Eigen::Vector3d> all_centroids(total_num_faces);
    std::vector<Eigen::Vector3d> all_normals(total_num_faces);
    std::vector<double> all_areas(total_num_faces);
    std::vector<double> all_temperatures(total_num_faces);
    std::vector<double> all_energy_accommodation_coefficients(total_num_faces);

    size_t face_idx = 0;
    int vertex_col_offset = 0;
    for (size_t i = 0; i < bodies.size(); ++i) {
        Eigen::Matrix3Xd rotated_vertices = bodies[i].vertices_B;
        std::vector<Eigen::Vector3d> rotated_centroids = bodies[i].centroids_B;
        std::vector<Eigen::Vector3d> rotated_normals = bodies[i].normals_B;

        rotateBody(rotated_vertices, rotated_centroids, rotated_normals,
                   bodies_rotation_angles_rad[i], bodies[i].rotation_direction_B, bodies[i].rotation_hinge_point_B);

        // Copy into preallocated buffer
        all_vertices.block(0, vertex_col_offset, 3, rotated_vertices.cols()) = rotated_vertices;
        vertex_col_offset += rotated_vertices.cols();

        for (size_t j = 0; j < bodies[i].centroids_B.size(); ++j) {
            all_centroids[face_idx] = rotated_centroids[j];
            all_normals[face_idx] = rotated_normals[j];
            all_areas[face_idx] = bodies[i].areas[j];
            all_temperatures[face_idx] = bodies[i].temperatures_K[j];
            all_energy_accommodation_coefficients[face_idx] = bodies[i].energy_accommodation_coefficients[j];
            face_idx++;
        }
    }

    Eigen::Vector3d v_rel_dir_B = v_rel_B.normalized();
    std::vector<bool> ind_not_shadowed = determineShadowedTriangles(all_vertices, all_centroids, all_normals, v_rel_dir_B);
    for (size_t i = 0; i < ind_not_shadowed.size(); ++i) {
        ind_not_shadowed[i] = !ind_not_shadowed[i];
    }

    // Reserve based on number of visible faces to reduce reallocations
    size_t visible_count = 0; for (bool b : ind_not_shadowed) if (b) ++visible_count;
    std::vector<double> areas_not_shadowed; areas_not_shadowed.reserve(visible_count);
    std::vector<Eigen::Vector3d> normals_not_shadowed; normals_not_shadowed.reserve(visible_count);
    std::vector<Eigen::Vector3d> centroids_not_shadowed; centroids_not_shadowed.reserve(visible_count);
    std::vector<Eigen::Vector3d> v_rels_not_shadowed; v_rels_not_shadowed.reserve(visible_count);
    std::vector<double> surface_temperatures_not_shadowed; surface_temperatures_not_shadowed.reserve(visible_count);
    std::vector<double> energy_accommodation_coefficients_not_shadowed; energy_accommodation_coefficients_not_shadowed.reserve(visible_count);

    for (size_t i = 0; i < total_num_faces; ++i) {
        if (ind_not_shadowed[i]) {
            areas_not_shadowed.push_back(all_areas[i]);
            normals_not_shadowed.push_back(all_normals[i]);
            centroids_not_shadowed.push_back(all_centroids[i]);
            surface_temperatures_not_shadowed.push_back(all_temperatures[i]);
            energy_accommodation_coefficients_not_shadowed.push_back(all_energy_accommodation_coefficients[i]);

            // Use cross product directly instead of forming a skew matrix
            Eigen::Vector3d v_indiv_B = v_rel_B - rotational_velocity_BI_B.cross(all_centroids[i]);
            v_rels_not_shadowed.push_back(v_indiv_B);
        }
    }

    return calcAeroForceAndTorque(areas_not_shadowed,
                                  normals_not_shadowed,
                                  centroids_not_shadowed,
                                  v_rels_not_shadowed,
                                  density,
                                  temperature,
                                  surface_temperatures_not_shadowed,
                                  energy_accommodation_coefficients_not_shadowed,
                                  particles_mass,
                                  temperature_ratio_method);
}

}
