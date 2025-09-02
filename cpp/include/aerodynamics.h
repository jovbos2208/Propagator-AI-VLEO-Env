#ifndef AERODYNAMICS_H
#define AERODYNAMICS_H

#include <Eigen/Dense>
#include <vector>

namespace vleo_aerodynamics_core {

struct AeroForceAndTorque {
    Eigen::Vector3d force;
    Eigen::Vector3d torque;
};

AeroForceAndTorque calcAeroForceAndTorque(const std::vector<double>& areas,
                                          const std::vector<Eigen::Vector3d>& normals,
                                          const std::vector<Eigen::Vector3d>& centroids,
                                          const std::vector<Eigen::Vector3d>& v_rels,
                                          double density,
                                          double gas_temperature,
                                          const std::vector<double>& surface_temperatures,
                                          const std::vector<double>& energy_accommodation_coefficients,
                                          double particles_mass,
                                          int temperature_ratio_method);

}

#endif // AERODYNAMICS_H
