#include "aerodynamics.h"
#include <iostream>
#include <cmath>

namespace vleo_aerodynamics_core {

const double KB = 1.38064852e-23; // Boltzmann constant

AeroForceAndTorque calcAeroForceAndTorque(const std::vector<double>& areas,
                                          const std::vector<Eigen::Vector3d>& normals,
                                          const std::vector<Eigen::Vector3d>& centroids,
                                          const std::vector<Eigen::Vector3d>& v_rels,
                                          const std::vector<double>& deltas,
                                          double density,
                                          double gas_temperature,
                                          const std::vector<double>& surface_temperatures,
                                          const std::vector<double>& energy_accommodation_coefficients,
                                          double particles_mass,
                                          int temperature_ratio_method) {

    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d total_torque = Eigen::Vector3d::Zero();

    double cm = sqrt(2.0 * KB * gas_temperature / particles_mass);

    for (size_t i = 0; i < areas.size(); ++i) {
        double V = v_rels[i].norm();
        double s = V / cm;
        double cosdelta = cos(deltas[i]);
        double scosdelta = s * cosdelta;

        double erfcterm = erfc(-scosdelta);
        double G1 = scosdelta / sqrt(M_PI) * exp(-scosdelta * scosdelta) + (0.5 + scosdelta * scosdelta) * erfcterm;
        double G2 = 1.0 / sqrt(M_PI) * exp(-scosdelta * scosdelta) + scosdelta * erfcterm;

        double T_rat;
        switch (temperature_ratio_method) {
            case 1: {
                double enum_ = scosdelta * erfcterm;
                double denom = 1.0 / sqrt(M_PI) * exp(-scosdelta * scosdelta) + enum_;
                T_rat = energy_accommodation_coefficients[i] * (2.0 * KB * surface_temperatures[i]) / (particles_mass * V * V) * s * s + (1.0 - energy_accommodation_coefficients[i]) * (1.0 + s * s / 2.0 + 0.25 * enum_ / denom);
                break;
            }
            case 2:
                T_rat = s * s / 2.0 * (1.0 + energy_accommodation_coefficients[i] * ((4.0 * KB * surface_temperatures[i]) / (particles_mass * V * V) - 1.0)) + 1.25 * (1.0 - energy_accommodation_coefficients[i]);
                break;
            case 3:
                T_rat = s * s / 2.0 * (1.0 + energy_accommodation_coefficients[i] * ((4.0 * KB * surface_temperatures[i]) / (particles_mass * V * V) - 1.0));
                break;
            default:
                throw std::runtime_error("Invalid temperature ratio method");
        }

        Eigen::Vector3d p = density / 2.0 * cm * cm * (-(G1 + sqrt(M_PI) / 2.0 * sqrt(T_rat) * G2) * normals[i] + s * G2 * (v_rels[i] / V + cosdelta * normals[i]));

        Eigen::Vector3d force = p * areas[i];
        Eigen::Vector3d torque = centroids[i].cross(force);

        total_force += force;
        total_torque += torque;
    }

    return {total_force, total_torque};
}

}
