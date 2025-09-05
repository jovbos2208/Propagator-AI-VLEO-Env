import numpy as np


def eci_to_rtn(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    r = r_eci / (np.linalg.norm(r_eci) + 1e-12)
    h = np.cross(r_eci, v_eci)
    w = h / (np.linalg.norm(h) + 1e-12)
    t = np.cross(w, r)
    R = np.stack([r, t, w], axis=1)  # columns are RTN axes in ECI
    return R


def target_quat(mode: str, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    # Nadir-pointing: body +Z to -R, +X along velocity, +Y completes RHS
    R = eci_to_rtn(r_eci, v_eci)
    # Construct LVLH frame in ECI: X along velocity (t-axis), Z to -r, Y completes
    x = R[:, 1]
    z = -R[:, 0]
    y = np.cross(z, x)
    C_be = np.stack([x, y, z], axis=1)  # body axes in ECI
    # Rotation matrix to quaternion (w,x,y,z)
    return rotm_to_quat(C_be)


def rotm_to_quat(C: np.ndarray) -> np.ndarray:
    m00, m01, m02 = C[0]
    m10, m11, m12 = C[1]
    m20, m21, m22 = C[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def quat_err(q_target: np.ndarray, q_be: np.ndarray) -> np.ndarray:
    # q_err = q_target * conj(q_be)
    w1, x1, y1, z1 = q_target
    w2, x2, y2, z2 = q_be
    w = w1 * w2 + x1 * (-x2) + y1 * (-y2) + z1 * (-z2)
    x = w1 * (-x2) + x1 * w2 + y1 * (-z2) - z1 * (-y2)
    y = w1 * (-y2) - x1 * (-z2) + y1 * w2 + z1 * (-x2)
    z = w1 * (-z2) + x1 * (-y2) - y1 * (-x2) + z1 * w2
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


# --- Earth-fixed conversions (approximate) ---

# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _gmst_rad(jd_utc: float) -> float:
    # Approximate GMST (IAU 1982) given Julian date UTC
    T = (jd_utc - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    gmst_rad = (gmst_sec % 86400.0) * (2.0 * np.pi / 86400.0)
    return gmst_rad


def eci_to_ecef(r_eci: np.ndarray, jd_utc: float) -> np.ndarray:
    # Rotate about Z by GMST to get ECEF
    theta = _gmst_rad(float(jd_utc))
    c, s = np.cos(theta), np.sin(theta)
    R3 = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return R3 @ np.asarray(r_eci, dtype=np.float64)


def ecef_to_geodetic(r_ecef: np.ndarray):
    # Convert ECEF (meters) to geodetic lat [rad], lon [rad], alt [m]
    x, y, z = map(float, r_ecef)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    if p < 1e-9:
        lat = np.sign(z) * (np.pi / 2)
        alt = abs(z) - _WGS84_A * np.sqrt(1.0 - _WGS84_E2)
        return lat, lon, alt
    # Bowring's method
    b = _WGS84_A * (1.0 - _WGS84_F)
    e2 = _WGS84_E2
    ep2 = ( _WGS84_A**2 - b**2 ) / (b**2)
    th = np.arctan2(_WGS84_A * z, b * p)
    lat = np.arctan2(z + ep2 * b * np.sin(th)**3, p - e2 * _WGS84_A * np.cos(th)**3)
    N = _WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return lat, lon, alt


# --- Solar position and eclipse (low-precision) ---

def _deg2rad(x: float) -> float:
    return float(x) * (np.pi / 180.0)


def sun_vec_eci(jd_utc: float) -> np.ndarray:
    """Approximate unit vector from Earth to Sun in ECI (equatorial) frame.

    Low-precision algorithm good to a few arcminutes. Based on NOAA/USNO formulas:
      - n = JD - J2000
      - L = 280.460 + 0.9856474 n (deg)
      - g = 357.528 + 0.9856003 n (deg)
      - lambda = L + 1.915 sin g + 0.020 sin 2g (deg)
      - epsilon = 23.439 - 0.0000004 n (deg)
      - ECI: [cos l, cos eps * sin l, sin eps * sin l]
    """
    n = float(jd_utc) - 2451545.0
    L = 280.460 + 0.9856474 * n
    g = 357.528 + 0.9856003 * n
    lam = L + 1.915 * np.sin(_deg2rad(g)) + 0.020 * np.sin(_deg2rad(2.0 * g))
    eps = 23.439 - 0.0000004 * n
    lam_r = _deg2rad(lam)
    eps_r = _deg2rad(eps)
    x = np.cos(lam_r)
    y = np.cos(eps_r) * np.sin(lam_r)
    z = np.sin(eps_r) * np.sin(lam_r)
    v = np.array([x, y, z], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def in_earth_eclipse(r_eci: np.ndarray, jd_utc: float, margin_m: float = 0.0) -> bool:
    """Cylindrical Earth shadow test.

    Returns True if the satellite at r_eci is behind Earth relative to the Sun and within
    the Earth's radius (plus margin) from the shadow axis.
    """
    s_hat = sun_vec_eci(jd_utc)
    r = np.asarray(r_eci, dtype=np.float64)
    # Dayside: dot > 0 -> illuminated
    if float(np.dot(r, s_hat)) > 0.0:
        return False
    # Perpendicular distance from Earth-Sun axis
    r_par = float(np.dot(r, s_hat))
    r_perp_vec = r - r_par * s_hat
    d = float(np.linalg.norm(r_perp_vec))
    R_E = 6371e3 + float(margin_m)
    return d < R_E
