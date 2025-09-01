#include "env_config.h"
#include <filesystem>

namespace vleo_aerodynamics_core {

static std::string resolve_example_path_impl(const std::string& name) {
    namespace fs = std::filesystem;
    fs::path base = fs::current_path();
    const std::vector<fs::path> rels = {
        fs::path("vleo-aerodynamics-c-export/vleo-aerodynamics-tool/example/obj_files") / name,
        fs::path("vleo-aerodynamics-c-export/vleo-aerodynamics-tool/example/gmsh_files") / name,
        fs::path("cpp/tests") / name,
        fs::path("tests") / name,
    };
    for (int up = 0; up < 6; ++up) {
        for (const auto& rel : rels) {
            fs::path cand = base / rel;
            if (fs::exists(cand)) return cand.string();
        }
        if (!base.has_parent_path()) break;
        base = base.parent_path();
    }
    return name; // best effort
}

static std::string choose_first_existing(const std::initializer_list<const char*>& names) {
    for (auto* n : names) {
        std::string p = resolve_example_path_impl(n);
        if (std::filesystem::exists(p)) return p;
    }
    // Return first resolved even if not existing (vleo-aero tools may create later)
    auto it = names.begin();
    return resolve_example_path_impl(*it);
}

GeometryConfig ShuttlecockDefaults::geometry() {
    GeometryConfig cfg;
    // Files order must match hinge/axis columns: [MainBody, WingRight, WingTop, WingLeft, WingBottom]
    cfg.object_files = {
        choose_first_existing({"MainBody.obj", "shuttle_box.obj", "box.obj", "cube.obj"}),
        choose_first_existing({"WingRight.obj", "wing_right.obj", "wing_pos_x.obj"}),
        choose_first_existing({"WingTop.obj", "wing_pos_y.obj"}),
        choose_first_existing({"WingLeft.obj", "wing_left.obj", "wing_neg_x.obj"}),
        choose_first_existing({"WingBottom.obj", "wing_neg_y.obj"}),
    };

    cfg.hinge_points_CAD.resize(3, 5);
    cfg.hinge_axes_CAD.resize(3, 5);
    cfg.hinge_points_CAD << 0.00,  0.05, 0.00, -0.05,  0.00,
                             0.00,  0.00, 0.05,  0.00, -0.05,
                             0.00,  0.00, 0.00,  0.00,  0.00;
    cfg.hinge_axes_CAD    << 1, 0, 1, 0, -1,
                             0, 1, 0, 1,  0,
                             0, 0, 0, 0,  0;

    cfg.temperatures_K = std::vector<std::vector<double>>(5, std::vector<double>{300.0});
    cfg.eac            = std::vector<std::vector<double>>(5, std::vector<double>{0.9});

    // Orientation and origin (same as example)
    cfg.DCM_B_from_CAD << 0, -1, 0,
                        -1, 0, 0,
                         0,  0, -1;
    cfg.CoM_CAD = Eigen::Vector3d::Zero();

    return cfg;
}

} // namespace vleo_aerodynamics_core
