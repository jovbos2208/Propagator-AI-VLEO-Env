#include "gravity.h"
#include <cmath>

namespace vleo_aerodynamics_core {

static inline void legendre_P_and_dP(double s, double& P2, double& dP2, double& P3, double& dP3, double& P4, double& dP4) {
    // P2(s) = 0.5(3 s^2 - 1);    P2'(s) = 3 s
    // P3(s) = 0.5(5 s^3 - 3 s);  P3'(s) = 0.5(15 s^2 - 3)
    // P4(s) = 1/8(35 s^4 - 30 s^2 + 3); P4'(s) = (35 s^3 - 15 s)/2
    const double s2 = s*s;
    const double s3 = s2*s;
    const double s4 = s2*s2;
    P2 = 0.5*(3.0*s2 - 1.0);
    dP2 = 3.0*s;
    P3 = 0.5*(5.0*s3 - 3.0*s);
    dP3 = 0.5*(15.0*s2 - 3.0);
    P4 = 0.125*(35.0*s4 - 30.0*s2 + 3.0);
    dP4 = 0.5*(35.0*s3 - 15.0*s);
}

Eigen::Vector3d gravity_accel_eci(const Eigen::Vector3d& r, const GravityParams& gp) {
    const double x = r.x();
    const double y = r.y();
    const double z = r.z();
    const double r2 = r.squaredNorm();
    const double r1 = std::sqrt(r2);
    const double r3 = r2 * r1;

    const double mu = gp.mu;
    const double Re = gp.Re;

    // Central term: -mu r / r^3
    Eigen::Vector3d a = -mu * r / r3;

    // Zonal harmonics J2–J4 via gradient of potential formulation
    if (gp.J2 != 0.0 || gp.J3 != 0.0 || gp.J4 != 0.0) {
        const double s = z / r1;           // sin(geocentric latitude)
        double P2, dP2, P3, dP3, P4, dP4;
        legendre_P_and_dP(s, P2, dP2, P3, dP3, P4, dP4);

        const double inv_r = 1.0 / r1;
        const double inv_r2 = 1.0 / r2;
        const double inv_r3 = 1.0 / r3;

        // f = 1 - sum Jn (Re/r)^n Pn(s)
        const double t2 = (Re*Re) * inv_r2;
        const double t3 = t2 * Re * inv_r;
        const double t4 = t2 * t2;
        const double f = 1.0
                       - gp.J2 * t2 * P2
                       - gp.J3 * t3 * P3
                       - gp.J4 * t4 * P4;

        // ∂f/∂x, ∂f/∂y, ∂f/∂z
        // Using: ∂g_n/∂x = Re^n x r^{-(n+3)} [ -n Pn(s) r - Pn'(s) z ]
        // and ∂g_n/∂z = Re^n r^{-(n+3)} [ -n z r Pn(s) + Pn'(s) (x^2 + y^2) ]
        const double Re2 = Re*Re;
        const double Re3 = Re2*Re;
        const double Re4 = Re2*Re2;
        const double inv_r4 = inv_r2 * inv_r2;
        const double inv_r5 = inv_r4 * inv_r;
        const double inv_r6 = inv_r3 * inv_r3;
        const double inv_r7 = inv_r6 * inv_r;
        const double xy2 = x*x + y*y;

        // ∂f/∂x and ∂f/∂y
        const double common2 = gp.J2 * Re2;
        const double common3 = gp.J3 * Re3;
        const double common4 = gp.J4 * Re4;

        const double dfx = x * (
            common2 * ( 2.0 * P2 * inv_r4 + dP2 * z * inv_r5 ) +
            common3 * ( 3.0 * P3 * inv_r5 + dP3 * z * inv_r6 ) +
            common4 * ( 4.0 * P4 * inv_r6 + dP4 * z * inv_r7 )
        );

        const double dfy = y * (
            common2 * ( 2.0 * P2 * inv_r4 + dP2 * z * inv_r5 ) +
            common3 * ( 3.0 * P3 * inv_r5 + dP3 * z * inv_r6 ) +
            common4 * ( 4.0 * P4 * inv_r6 + dP4 * z * inv_r7 )
        );

        // ∂f/∂z
        const double dfz =
            common2 * ( 2.0 * z * P2 * inv_r4 - dP2 * xy2 * inv_r5 ) +
            common3 * ( 3.0 * z * P3 * inv_r5 - dP3 * xy2 * inv_r6 ) +
            common4 * ( 4.0 * z * P4 * inv_r6 - dP4 * xy2 * inv_r7 );

        // a = mu [ (∂f/∂x)/r - f x / r^3, ... ]
        a.x() = mu * (dfx * inv_r - f * x * inv_r3);
        a.y() = mu * (dfy * inv_r - f * y * inv_r3);
        a.z() = mu * (dfz * inv_r - f * z * inv_r3);
    }

    return a;
}

} // namespace vleo_aerodynamics_core
