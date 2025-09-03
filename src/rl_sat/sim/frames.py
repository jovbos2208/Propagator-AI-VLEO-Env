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

