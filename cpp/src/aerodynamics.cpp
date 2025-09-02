#include "aerodynamics.h"
#include <iostream>
#include <cmath>

namespace vleo_aerodynamics_core {

const double KB = 1.38064852e-23; // Boltzmann constant

AeroForceAndTorque calcAeroForceAndTorque(const std::vector<double>& areas,
                                          const std::vector<Eigen::Vector3d>& normals,
                                          const std::vector<Eigen::Vector3d>& centroids,
                                          const std::vector<Eigen::Vector3d>& v_rels,
                                          double density,
                                          double gas_temperature,
                                          const std::vector<double>& surface_temperatures,
                                          const std::vector<double>& energy_accommodation_coefficients,
                                          double particles_mass,
                                          int temperature_ratio_method) {

    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d total_torque = Eigen::Vector3d::Zero();

    double cm = sqrt(2.0 * KB * gas_temperature / particles_mass);

    const double inv_sqrt_pi = 1.0 / std::sqrt(M_PI);
    for (size_t i = 0; i < areas.size(); ++i) {
        double V = v_rels[i].norm();
        if (V <= 1e-12) {
            continue; // no relative flow => no aero force
        }
        double s = V / cm;
        // cos(delta) = -v_hat dot n
        Eigen::Vector3d vhat = v_rels[i] / V;
        double cosdelta = -vhat.dot(normals[i]);
        double scosdelta = s * cosdelta;

        double e = std::exp(-scosdelta * scosdelta);
        double erfcterm = std::erfc(-scosdelta);
        double G1 = scosdelta * inv_sqrt_pi * e + (0.5 + scosdelta * scosdelta) * erfcterm;
        double G2 = inv_sqrt_pi * e + scosdelta * erfcterm;

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

        Eigen::Vector3d p = density / 2.0 * cm * cm * (-(G1 + std::sqrt(M_PI) / 2.0 * std::sqrt(T_rat) * G2) * normals[i] + s * G2 * (vhat + cosdelta * normals[i]));

        Eigen::Vector3d force = p * areas[i];
        Eigen::Vector3d torque = centroids[i].cross(force);

        total_force += force;
        total_torque += torque;
    }

    return {total_force, total_torque};
}

}
