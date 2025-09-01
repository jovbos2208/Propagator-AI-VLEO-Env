#include "shadowing.h"
#include <iostream>
#include <cmath>

namespace vleo_aerodynamics_core {

static int sgn(double x) { return (x > 0) - (x < 0); }

static bool checkOriginInAnyTriangle(const std::vector<Eigen::MatrixXd>& vertices_proj) {
    for (const auto& triangle_vertices : vertices_proj) {
        Eigen::Vector2d a = triangle_vertices.col(0);
        Eigen::Vector2d b = triangle_vertices.col(1);
        Eigen::Vector2d c = triangle_vertices.col(2);

        Eigen::Vector2d v0 = b - a;
        Eigen::Vector2d v1 = c - a;
        Eigen::Vector2d v2 = -a;

        double d00 = v0.dot(v0);
        double d01 = v0.dot(v1);
        double d11 = v1.dot(v1);
        double d20 = v2.dot(v0);
        double d21 = v2.dot(v1);

        double invDenom = 1.0 / (d00 * d11 - d01 * d01);
        double u = (d11 * d20 - d01 * d21) * invDenom;
        double v = (d00 * d21 - d01 * d20) * invDenom;

        if ((u >= 0) && (v >= 0) && (u + v <= 1)) {
            return true;
        }
    }
    return false;
}

std::vector<bool> determineShadowedTriangles(const Eigen::Matrix3Xd& vertices,
                                             const std::vector<Eigen::Vector3d>& centroids,
                                             const std::vector<Eigen::Vector3d>& normals,
                                             const Eigen::Vector3d& dir) {
    int num_triangles = vertices.cols() / 3;
    if (vertices.cols() % 3 != 0) {
        // Handle error: vertices matrix does not contain a whole number of triangles
        throw std::runtime_error("Vertices matrix does not contain a whole number of triangles.");
    }
    std::vector<bool> ind_shadowed(num_triangles, false);

    // 1st Reduction
    std::vector<bool> ind_shadowable(num_triangles);
    std::vector<bool> ind_shadowing(num_triangles);
    for (int i = 0; i < num_triangles; ++i) {
        double delta = acos(-dir.dot(normals[i]));
        ind_shadowable[i] = (delta <= M_PI / 2.0);
        ind_shadowing[i] = !ind_shadowable[i];
    }

    // 2nd Reduction
    Eigen::Matrix3Xd vertices_list(3, 3 * num_triangles);
    for (int i = 0; i < num_triangles; ++i) {
        vertices_list.block(0, 3 * i, 3, 3) = vertices.block(0, 3 * i, 3, 3);
    }

    Eigen::RowVectorXd w_list = dir.transpose() * vertices_list; // 1 x (3*num_triangles)
    Eigen::MatrixXd w(3, num_triangles);
    for (int i = 0; i < num_triangles; ++i) {
        w.col(i) = w_list.segment(3 * i, 3).transpose();
    }

    double max_w_shadowable = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < num_triangles; ++i) {
        if (ind_shadowable[i]) {
            max_w_shadowable = std::max(max_w_shadowable, w.col(i).maxCoeff());
        }
    }

    double min_w_shadowing = std::numeric_limits<double>::infinity();
    for (int i = 0; i < num_triangles; ++i) {
        if (ind_shadowing[i]) {
            min_w_shadowing = std::min(min_w_shadowing, w.col(i).minCoeff());
        }
    }

    for (int i = 0; i < num_triangles; ++i) {
        ind_shadowing[i] = ind_shadowing[i] && (w.col(i).array() < max_w_shadowable).any();
        ind_shadowable[i] = ind_shadowable[i] && (w.col(i).array() > min_w_shadowing).any();
    }

    // Build two orthonormal axes perpendicular to dir
    Eigen::Vector3d u = dir.normalized();
    // choose a helper vector not parallel to u
    Eigen::Vector3d helper;
    if (std::abs(u.x()) <= std::abs(u.y()) && std::abs(u.x()) <= std::abs(u.z())) {
        helper = Eigen::Vector3d::UnitX();
    } else if (std::abs(u.y()) <= std::abs(u.x()) && std::abs(u.y()) <= std::abs(u.z())) {
        helper = Eigen::Vector3d::UnitY();
    } else {
        helper = Eigen::Vector3d::UnitZ();
    }
    Eigen::Vector3d v = (helper - u * (u.dot(helper)));
    double vnorm = v.norm();
    if (vnorm == 0.0) {
        // fallback: pick another helper
        helper = (helper == Eigen::Vector3d::UnitX()) ? Eigen::Vector3d::UnitY() : Eigen::Vector3d::UnitX();
        v = (helper - u * (u.dot(helper))); vnorm = v.norm();
        if (vnorm == 0.0) {
            throw std::runtime_error("Cannot construct perpendicular basis for direction vector.");
        }
    }
    v /= vnorm;
    Eigen::Vector3d w_perp = u.cross(v);
    Eigen::Matrix<double,3,2> perp_axes; perp_axes.col(0) = v; perp_axes.col(1) = w_perp;

    for (int i = 0; i < num_triangles; ++i) {
        if (ind_shadowable[i]) {
            // 3rd Reduction
            std::vector<int> current_shadowing_indices;
            for (int j = 0; j < num_triangles; ++j) {
                if (ind_shadowing[j]) {
                    if (normals[j].dot(centroids[i] - centroids[j]) >= 0) {
                        current_shadowing_indices.push_back(j);
                    }
                }
            }

            if (!current_shadowing_indices.empty()) {
                std::vector<Eigen::MatrixXd> vertices_proj;
                for (int j : current_shadowing_indices) {
                    Eigen::MatrixXd temp_vertices_centered = vertices.block(0, 3 * j, 3, 3);
                    temp_vertices_centered.colwise() -= centroids[i];
                    Eigen::MatrixXd proj_vertices = perp_axes.transpose() * temp_vertices_centered; // 2x3
                    if (proj_vertices.rows() != 2 || proj_vertices.cols() != 3) {
                        throw std::runtime_error("Projected vertices matrix has unexpected dimensions.");
                    }
                    vertices_proj.push_back(proj_vertices);
                }

                // 4th Reduction
                std::vector<Eigen::MatrixXd> remaining_vertices_proj;
                for (const auto& proj_verts : vertices_proj) {
                    int sx0 = sgn(proj_verts(0, 0));
                    int sx1 = sgn(proj_verts(0, 1));
                    int sx2 = sgn(proj_verts(0, 2));
                    int sumx = sx0 + sx1 + sx2;
                    bool sign_change_x = std::abs(sumx) < 3;

                    int sy0 = sgn(proj_verts(1, 0));
                    int sy1 = sgn(proj_verts(1, 1));
                    int sy2 = sgn(proj_verts(1, 2));
                    int sumy = sy0 + sy1 + sy2;
                    bool sign_change_y = std::abs(sumy) < 3;

                    if (sign_change_x && sign_change_y) {
                        remaining_vertices_proj.push_back(proj_verts);
                    }
                }

                if (!remaining_vertices_proj.empty()) {
                    if (checkOriginInAnyTriangle(remaining_vertices_proj)) {
                        ind_shadowed[i] = true;
                    }
                }
            }
        }
    }

    return ind_shadowed;
}

}
