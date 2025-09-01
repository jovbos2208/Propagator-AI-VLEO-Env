#ifndef SHADOWING_H
#define SHADOWING_H

#include <Eigen/Dense>
#include <vector>

namespace vleo_aerodynamics_core {

std::vector<bool> determineShadowedTriangles(const Eigen::Matrix3Xd& vertices,
                                             const std::vector<Eigen::Vector3d>& centroids,
                                             const std::vector<Eigen::Vector3d>& normals,
                                             const Eigen::Vector3d& dir);

}

#endif // SHADOWING_H
