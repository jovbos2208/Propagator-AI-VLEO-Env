#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include "dynamics.h"

namespace vleo_aerodynamics_core {

// Single RK4 step for our State
void rk4_step(State& x,
              const Controls& u,
              const Params& p,
              const AtmoSample& atmo,
              AeroAdapter& aero,
              double dt);

struct DP54Params {
    double atol = 1e-6;
    double rtol = 1e-6;
    double safety = 0.9;
    double min_dt = 1e-4;
    double max_dt = 10.0;
};

// Perform one adaptive Dormand–Prince(5,4) step. Returns whether step accepted.
// On success, advances state and suggests next dt via dt_out; on failure, dt_out is reduced.
bool dp54_adaptive_step(State& x,
                        const Controls& u,
                        const Params& p,
                        const AtmoSample& atmo,
                        AeroAdapter& aero,
                        double& dt_inout,
                        const DP54Params& prm);

} // namespace vleo_aerodynamics_core

#endif // INTEGRATORS_H
