#ifndef SHOW_BODIES_H
#define SHOW_BODIES_H

#include "body_importer.h"
#include <string>
#include <vector>

namespace vleo_aerodynamics_core {

// Writes a combined OBJ of all bodies rotated by the given angles.
// Each body is exported as a separate object group.
void showBodies(const std::vector<Body>& bodies,
                const std::vector<double>& bodies_rotation_angles_rad,
                const std::string& out_obj_path);

}

#endif // SHOW_BODIES_H

