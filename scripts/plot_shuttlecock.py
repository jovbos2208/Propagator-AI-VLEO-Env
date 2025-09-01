#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import math as _math

def detect_schema(fieldnames):
    f = set(fieldnames or [])
    # Prefer explicit AoA/AoS schema if available
    if {'aoa_deg','aos_deg','Fx','Fy','Fz'} <= f:
        return 'aoa_aos'
    if {'eta1_deg','eta2_deg','aoa_deg'} <= f:
        return 'grid'
    if {'attitude_idx','surface_idx','angle_deg'} <= f:
        return 'envelope'
    return 'unknown'

def load_csv_grid(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'eta1': int(float(row['eta1_deg'])),
                'eta2': int(float(row['eta2_deg'])),
                'aoa': int(float(row['aoa_deg'])),
                'Fx': float(row['Fx']),
                'Fy': float(row['Fy']),
                'Fz': float(row['Fz']),
                'Tx': float(row['Tx']),
                'Ty': float(row['Ty']),
                'Tz': float(row['Tz']),
            })
    return rows

def load_csv_envelope(path):
    # Returns dict[(attitude,surface)] -> list of points sorted by angle
    data = defaultdict(list)
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            att = int(float(row['attitude_idx']))
            surf = int(float(row['surface_idx']))
            ang = float(row['angle_deg'])
            data[(att, surf)].append({
                'angle': ang,
                'Fx': float(row['Fx']),
                'Fy': float(row['Fy']),
                'Fz': float(row['Fz']),
                'Tx': float(row['Tx']),
                'Ty': float(row['Ty']),
                'Tz': float(row['Tz']),
            })
    for key in data:
        data[key].sort(key=lambda d: d['angle'])
    return data

def load_csv_aoa_aos(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'aoa': float(row['aoa_deg']),
                'aos': float(row['aos_deg']),
                'Vx': float(row.get('Vx', 0.0)),
                'Vy': float(row.get('Vy', 0.0)),
                'Vz': float(row.get('Vz', 0.0)),
                'rho': float(row.get('rho', 'nan')),
                'T': float(row.get('T_K', 'nan')),
                'mass': float(row.get('mass', 'nan')),
                'method': float(row.get('method', 'nan')),
                's': float(row.get('s', 'nan')),
                'eta1': float(row.get('eta1_deg', 'nan')),
                'eta2': float(row.get('eta2_deg', 'nan')),
                'Fx': float(row['Fx']),
                'Fy': float(row['Fy']),
                'Fz': float(row['Fz']),
                'Tx': float(row['Tx']),
                'Ty': float(row['Ty']),
                'Tz': float(row['Tz']),
            })
    rows.sort(key=lambda r: (r['aos'], r['aoa']))
    return rows

def slice_aoa(rows, eta1=0, eta2=0):
    s = [r for r in rows if r['eta1']==eta1 and r['eta2']==eta2]
    s.sort(key=lambda r: r['aoa'])
    return s

def slice_eta1(rows, aoa=0, eta2=0):
    s = [r for r in rows if r['aoa']==aoa and r['eta2']==eta2]
    s.sort(key=lambda r: r['eta1'])
    return s

def slice_eta2(rows, aoa=0, eta1=0):
    s = [r for r in rows if r['aoa']==aoa and r['eta1']==eta1]
    s.sort(key=lambda r: r['eta2'])
    return s

def heatmap(rows, aoa):
    # Build |F| map over eta1 x eta2 for a given aoa
    data = defaultdict(dict)
    eta1_vals = set()
    eta2_vals = set()
    for r in rows:
        if r['aoa'] != aoa: continue
        eta1_vals.add(r['eta1'])
        eta2_vals.add(r['eta2'])
        magF = math.sqrt(r['Fx']**2 + r['Fy']**2 + r['Fz']**2)
        magT = math.sqrt(r['Tx']**2 + r['Ty']**2 + r['Tz']**2)
        data[r['eta1']][r['eta2']] = (magF, magT)
    e1 = sorted(eta1_vals)
    e2 = sorted(eta2_vals)
    F = np.zeros((len(e2), len(e1)))
    T = np.zeros_like(F)
    for i, et2 in enumerate(e2):
        for j, et1 in enumerate(e1):
            F[i, j] = data[et1][et2][0]
            T[i, j] = data[et1][et2][1]
    return e1, e2, F, T

def plot_envelopes(env, outdir):
    # env: dict[(attitude,surface)] -> list of dict with keys angle, Fx..Tz
    attitudes = sorted(set(att for att, _ in env.keys()))
    surfaces = sorted(set(s for _, s in env.keys()))
    for att in attitudes:
        for surf in surfaces:
            seq = env.get((att, surf), [])
            if not seq:
                continue
            angles = [p['angle'] for p in seq]
            # Forces
            plt.figure()
            for comp in ['Fx','Fy','Fz']:
                plt.plot(angles, [p[comp] for p in seq], label=comp)
            plt.xlabel('angle (deg)'); plt.ylabel('Force (N)')
            plt.title(f'Forces vs angle (att={att}, surface={surf})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(f"{outdir}/force_vs_angle_att{att}_surf{surf}.png"); plt.clf()
            # Torques
            plt.figure()
            for comp in ['Tx','Ty','Tz']:
                plt.plot(angles, [p[comp] for p in seq], label=comp)
            plt.xlabel('angle (deg)'); plt.ylabel('Torque (Nm)')
            plt.title(f'Torques vs angle (att={att}, surface={surf})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(f"{outdir}/torque_vs_angle_att{att}_surf{surf}.png"); plt.clf()

def plot_aoa_aos(rows, outdir):
    import numpy as _np
    aoas = sorted({r['aoa'] for r in rows})
    aoses = sorted({r['aos'] for r in rows})
    # If degenerate (single AoS), plot lines vs AoA
    if len(aoses) == 1:
        x = [r['aoa'] for r in rows]
        for comp in ['Fx','Fy','Fz']:
            plt.plot(x, [r[comp] for r in rows], label=comp)
        plt.xlabel('AoA (deg)'); plt.ylabel('Force (N)'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/force_vs_aoa.png"); plt.clf()
        for comp in ['Tx','Ty','Tz']:
            plt.plot(x, [r[comp] for r in rows], label=comp)
        plt.xlabel('AoA (deg)'); plt.ylabel('Torque (Nm)'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/torque_vs_aoa.png"); plt.clf()
        # Magnitudes
        Fmag = [_math.sqrt(r['Fx']**2 + r['Fy']**2 + r['Fz']**2) for r in rows]
        Tmag = [_math.sqrt(r['Tx']**2 + r['Ty']**2 + r['Tz']**2) for r in rows]
        plt.plot(x, Fmag); plt.xlabel('AoA (deg)'); plt.ylabel('|F| (N)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/Fmag_vs_aoa.png"); plt.clf()
        plt.plot(x, Tmag); plt.xlabel('AoA (deg)'); plt.ylabel('|T| (Nm)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/Tmag_vs_aoa.png"); plt.clf()
        return
    # Otherwise build heatmaps over AoA x AoS
    A = len(aoas); B = len(aoses)
    Fmag = _np.zeros((B, A)); Tmag = _np.zeros_like(Fmag)
    idx = {(r['aoa'], r['aos']): r for r in rows}
    for i, aos in enumerate(aoses):
        for j, aoa in enumerate(aoas):
            r = idx[(aoa, aos)]
            Fmag[i, j] = _math.sqrt(r['Fx']**2 + r['Fy']**2 + r['Fz']**2)
            Tmag[i, j] = _math.sqrt(r['Tx']**2 + r['Ty']**2 + r['Tz']**2)
    plt.imshow(Fmag, origin='lower', extent=[aoas[0], aoas[-1], aoses[0], aoses[-1]], aspect='auto')
    plt.colorbar(label='|F| (N)'); plt.xlabel('AoA (deg)'); plt.ylabel('AoS (deg)')
    plt.title('|F| over AoA/AoS'); plt.tight_layout(); plt.savefig(f"{outdir}/Fmag_aoa_aos.png"); plt.clf()
    plt.imshow(Tmag, origin='lower', extent=[aoas[0], aoas[-1], aoses[0], aoses[-1]], aspect='auto')
    plt.colorbar(label='|T| (Nm)'); plt.xlabel('AoA (deg)'); plt.ylabel('AoS (deg)')
    plt.title('|T| over AoA/AoS'); plt.tight_layout(); plt.savefig(f"{outdir}/Tmag_aoa_aos.png"); plt.clf()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='cpp/build/sweep_results.csv')
    ap.add_argument('--outdir', default='cpp/build')
    args = ap.parse_args()

    # Peek header to detect schema
    with open(args.csv, 'r') as f:
        r = csv.reader(f)
        header = next(r)
    schema = detect_schema(header)

    if schema == 'grid':
        rows = load_csv_grid(args.csv)
    elif schema == 'envelope':
        env = load_csv_envelope(args.csv)
    elif schema == 'aoa_aos':
        rows_ao = load_csv_aoa_aos(args.csv)
    else:
        raise RuntimeError(f"Unknown CSV schema: {header}")

    # Draw a simple top-view representation of the assembled body for eta1=50, eta2=50, AoA=0
    def rot2d(px, py, cx, cy, ang):
        ca, sa = _math.cos(ang), _math.sin(ang)
        x, y = px - cx, py - cy
        return cx + ca*x - sa*y, cy + sa*x + ca*y

    def draw_body(ax, eta1_deg=50, eta2_deg=50, aoa_deg=0):
        eta1 = _math.radians(eta1_deg)
        eta2 = _math.radians(eta2_deg)
        # Box footprint
        bx = [-0.05, 0.05, 0.05, -0.05]
        by = [-0.05, -0.05, 0.05, 0.05]
        ax.add_patch(Polygon(np.c_[bx,by], closed=True, facecolor=(0.8,0.8,0.8), edgecolor='k'))
        # Wings (top view rectangles), pre-rotation footprints (thin along one axis)
        wings = []
        # +X wing, hinge at (0.05, 0)
        wings.append(([(0.049,-0.049),(0.050,-0.049),(0.050,0.049),(0.049,0.049)], (0.05,0.0), +eta1))
        # -X wing, hinge at (-0.05, 0)
        wings.append(( [(-0.050,-0.049),(-0.049,-0.049),(-0.049,0.049),(-0.050,0.049)], (-0.05,0.0), -eta1))
        # +Y wing, hinge at (0, 0.05)
        wings.append(( [(-0.049,0.049),(-0.049,0.050),(0.049,0.050),(0.049,0.049)], (0.0,0.05), +eta2))
        # -Y wing, hinge at (0, -0.05)
        wings.append(( [(-0.049,-0.050),(-0.049,-0.049),(0.049,-0.049),(0.049,-0.050)], (0.0,-0.05), -eta2))
        for pts, hinge, ang in wings:
            cx, cy = hinge
            rot_pts = [rot2d(x,y,cx,cy,ang) for (x,y) in pts]
            ax.add_patch(Polygon(np.array(rot_pts), closed=True, facecolor=(0.6,0.8,1.0), edgecolor='k'))
        # AoA arrow
        aoa = _math.radians(aoa_deg)
        Vx, Vy = _math.cos(aoa), _math.sin(aoa)
        ax.arrow(-0.12, -0.12, 0.08*Vx, 0.08*Vy, head_width=0.01, head_length=0.02, fc='r', ec='r')
        ax.text(-0.12, -0.14, f"AoA={aoa_deg} deg", color='r')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Assembled body (eta1={eta1_deg}°, eta2={eta2_deg}°)')
        ax.grid(True)

    # Always draw a simple assembled body graphic
    fig, ax = plt.subplots(figsize=(5,5))
    draw_body(ax, 50, 50, 0)
    plt.tight_layout(); plt.savefig(f"{args.outdir}/assembled_body.png"); plt.clf()

    if schema == 'grid':
        # Line plots
        for label, sfun, xkey in [
            ('aoa', slice_aoa, 'aoa'),
            ('eta1', slice_eta1, 'eta1'),
            ('eta2', slice_eta2, 'eta2'),
        ]:
            if label == 'aoa': s = sfun(rows, eta1=0, eta2=0)
            elif label == 'eta1': s = sfun(rows, aoa=0, eta2=0)
            else: s = sfun(rows, aoa=0, eta1=0)
            x = [r[xkey] for r in s]
            for comp, key in [('Fx','Fx'),('Fy','Fy'),('Fz','Fz')]:
                y = [r[key] for r in s]
                plt.plot(x, y, label=comp)
            plt.xlabel(label + ' (deg)')
            plt.ylabel('Force (N)')
            plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.outdir}/force_vs_{label}.png")
            plt.clf()

            for comp, key in [('Tx','Tx'),('Ty','Ty'),('Tz','Tz')]:
                y = [r[key] for r in s]
                plt.plot(x, y, label=comp)
            plt.xlabel(label + ' (deg)')
            plt.ylabel('Torque (Nm)')
            plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.outdir}/torque_vs_{label}.png")
            plt.clf()

        # Heatmaps at AoA slices
        for aoa in [-15, 0, 15]:
            e1, e2, F, T = heatmap(rows, aoa)
            plt.imshow(F, origin='lower', extent=[e1[0], e1[-1], e2[0], e2[-1]], aspect='auto')
            plt.colorbar(label='|F| (N)')
            plt.xlabel('eta1 (deg)'); plt.ylabel('eta2 (deg)')
            plt.title(f'|F| heatmap at AoA={aoa} deg')
            plt.tight_layout(); plt.savefig(f"{args.outdir}/F_heatmap_aoa_{aoa}.png"); plt.clf()

            plt.imshow(T, origin='lower', extent=[e1[0], e1[-1], e2[0], e2[-1]], aspect='auto')
            plt.colorbar(label='|T| (Nm)')
            plt.xlabel('eta1 (deg)'); plt.ylabel('eta2 (deg)')
            plt.title(f'|T| heatmap at AoA={aoa} deg')
            plt.tight_layout(); plt.savefig(f"{args.outdir}/T_heatmap_aoa_{aoa}.png"); plt.clf()
    elif schema == 'envelope':
        plot_envelopes(env, args.outdir)
    else:  # aoa_aos
        plot_aoa_aos(rows_ao, args.outdir)

    print('Saved plots to', args.outdir)

if __name__ == '__main__':
    main()
