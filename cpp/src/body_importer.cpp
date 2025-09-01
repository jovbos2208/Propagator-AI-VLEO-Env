#include "body_importer.h"
#include <iostream>

namespace vleo_aerodynamics_core {

std::vector<Body> importMultipleBodies(const std::vector<std::string>& object_files,
                                       const Eigen::Matrix3Xd& rotation_hinge_points_CAD,
                                       const Eigen::Matrix3Xd& rotation_directions_CAD,
                                       const std::vector<std::vector<double>>& temperatures_K,
                                       const std::vector<std::vector<double>>& energy_accommodation_coefficients,
                                       const Eigen::Matrix3d& DCM_B_from_CAD,
                                       const Eigen::Vector3d& CoM_CAD) {

    if (abs(DCM_B_from_CAD.determinant() - 1.0) > 1e-6) {
        // The determinant of DCM_B_from_CAD should be 1.0
        // You might want to handle this error more gracefully
        std::cerr << "Warning: Determinant of DCM is not equal to 1. The DCM will be scaled accordingly." << std::endl;
    }

    int num_bodies = object_files.size();
    std::vector<Body> bodies(num_bodies);

    for (int i = 0; i < num_bodies; ++i) {
        ObjData obj_data = loadObjFile(object_files[i]);

        // Transform vertices to body frame
        obj_data.vertices = DCM_B_from_CAD * (obj_data.vertices.colwise() - CoM_CAD);

        bodies[i].vertices_B = obj_data.vertices;
        bodies[i].centroids_B = obj_data.centroids;
        bodies[i].normals_B = obj_data.normals;
        bodies[i].areas = obj_data.areas;

        // Transform rotation hinge point and direction to body frame
        bodies[i].rotation_hinge_point_B = DCM_B_from_CAD * (rotation_hinge_points_CAD.col(i) - CoM_CAD);
        bodies[i].rotation_direction_B = DCM_B_from_CAD * rotation_directions_CAD.col(i);

        // Set temperatures and energy accommodation coefficients
        bodies[i].temperatures_K.resize(obj_data.vertices.cols() / 3);
        if (temperatures_K[i].size() == 1) {
            std::fill(bodies[i].temperatures_K.begin(), bodies[i].temperatures_K.end(), temperatures_K[i][0]);
        } else {
            bodies[i].temperatures_K = temperatures_K[i];
        }

        bodies[i].energy_accommodation_coefficients.resize(obj_data.vertices.cols() / 3);
        if (energy_accommodation_coefficients[i].size() == 1) {
            std::fill(bodies[i].energy_accommodation_coefficients.begin(), bodies[i].energy_accommodation_coefficients.end(), energy_accommodation_coefficients[i][0]);
        } else {
            bodies[i].energy_accommodation_coefficients = energy_accommodation_coefficients[i];
        }
    }

    return bodies;
}

}
