#!/usr/bin/env python3
import csv
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused, enables 3D


def load_envelope(csv_path):
    groups = {}
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            att = int(float(row['attitude_idx']))
            surf = int(float(row['surface_idx']))
            ang = float(row['angle_deg'])
            key = (att, surf)
            groups.setdefault(key, []).append({
                'angle': ang,
                'Fx': float(row['Fx']),
                'Fy': float(row['Fy']),
                'Fz': float(row['Fz']),
                'Tx': float(row['Tx']),
                'Ty': float(row['Ty']),
                'Tz': float(row['Tz']),
            })
    # sort by angle
    for key in groups:
        groups[key].sort(key=lambda d: d['angle'])
    return groups


def plot_lines(groups, outdir, filter_att=None, filter_surf=None):
    os.makedirs(outdir, exist_ok=True)
    for (att, surf), seq in groups.items():
        if filter_att is not None and att != filter_att:
            continue
        if filter_surf is not None and surf != filter_surf:
            continue
        x = [p['angle'] for p in seq]
        # Forces
        plt.figure()
        for comp in ['Fx', 'Fy', 'Fz']:
            plt.plot(x, [p[comp] for p in seq], label=comp)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Force (N)')
        plt.title(f'Forces vs angle (att={att}, surface={surf})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'force_vs_angle_att{att}_surf{surf}.png'))
        plt.close()
        
        # Torques
        plt.figure()
        for comp in ['Tx', 'Ty', 'Tz']:
            plt.plot(x, [p[comp] for p in seq], label=comp)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Torque (Nm)')
        plt.title(f'Torques vs angle (att={att}, surface={surf})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'torque_vs_angle_att{att}_surf{surf}.png'))
        plt.close()


def plot_3d_envelopes(groups, outdir):
    # Create 3D envelope plots akin to MATLAB's example.m per attitude
    # Collect attitudes and surfaces
    attitudes = sorted(set(att for (att, _surf) in groups.keys()))
    surfaces = sorted(set(surf for (_att, surf) in groups.keys()))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for att in attitudes:
        # Force envelopes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx, surf in enumerate(surfaces):
            seq = groups.get((att, surf), [])
            if not seq:
                continue
            Fx = [p['Fx'] for p in seq]
            Fy = [p['Fy'] for p in seq]
            Fz = [p['Fz'] for p in seq]
            ax.plot(Fx, Fy, Fz, color=colors[idx % len(colors)], label=f'Surface {surf}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Individual Force Envelopes')
        try:
            # Rough equal aspect
            import numpy as _np
            pts = []
            for (_k, _s), seq in groups.items():
                if _k == att and seq:
                    pts.extend([(p['Fx'], p['Fy'], p['Fz']) for p in seq])
            if pts:
                P = _np.array(pts)
                ranges = P.max(axis=0) - P.min(axis=0)
                ranges[ranges == 0] = 1.0
                ax.set_box_aspect(ranges)
        except Exception:
            pass
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'force_envelopes_att{att}.png'))
        plt.close()

        # Torque envelopes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx, surf in enumerate(surfaces):
            seq = groups.get((att, surf), [])
            if not seq:
                continue
            Tx = [p['Tx'] for p in seq]
            Ty = [p['Ty'] for p in seq]
            Tz = [p['Tz'] for p in seq]
            ax.plot(Tx, Ty, Tz, color=colors[idx % len(colors)], label=f'Surface {surf}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Individual Torque Envelopes')
        try:
            import numpy as _np
            pts = []
            for (_k, _s), seq in groups.items():
                if _k == att and seq:
                    pts.extend([(p['Tx'], p['Ty'], p['Tz']) for p in seq])
            if pts:
                P = _np.array(pts)
                ranges = P.max(axis=0) - P.min(axis=0)
                ranges[ranges == 0] = 1.0
                ax.set_box_aspect(ranges)
        except Exception:
            pass
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'torque_envelopes_att{att}.png'))
        plt.close()

def main():
    ap = argparse.ArgumentParser(description='Plot force/torque components vs angle from example_envelope.csv')
    ap.add_argument('--csv', default='example_envelope.csv')
    ap.add_argument('--total-csv', default='example_envelope_total.csv')
    ap.add_argument('--outdir', default='cpp/build')
    ap.add_argument('--att', type=int, default=None, help='Filter by attitude index (1..N)')
    ap.add_argument('--surf', type=int, default=None, help='Filter by surface index (1..4)')
    args = ap.parse_args()

    groups = load_envelope(args.csv)
    plot_lines(groups, args.outdir, args.att, args.surf)
    plot_3d_envelopes(groups, args.outdir)
    # Plot totals if available
    if args.total_csv and os.path.exists(args.total_csv):
        totals = {}
        with open(args.total_csv, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                att = int(float(row['attitude_idx']))
                ang = float(row['angle_deg'])
                totals.setdefault(att, []).append({
                    'angle': ang,
                    'Fx': float(row['Fx']), 'Fy': float(row['Fy']), 'Fz': float(row['Fz']),
                    'Tx': float(row['Tx']), 'Ty': float(row['Ty']), 'Tz': float(row['Tz']),
                })
        for att, seq in totals.items():
            seq.sort(key=lambda d: d['angle'])
            x = [p['angle'] for p in seq]
            # Forces total
            plt.figure()
            for comp in ['Fx','Fy','Fz']:
                plt.plot(x, [p[comp] for p in seq], label=comp)
            plt.xlabel('Angle (deg)'); plt.ylabel('Force (N)'); plt.title(f'Total Forces vs angle (att={att})')
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'total_force_vs_angle_att{att}.png'))
            plt.close()
            # Torques total
            plt.figure()
            for comp in ['Tx','Ty','Tz']:
                plt.plot(x, [p[comp] for p in seq], label=comp)
            plt.xlabel('Angle (deg)'); plt.ylabel('Torque (Nm)'); plt.title(f'Total Torques vs angle (att={att})')
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'total_torque_vs_angle_att{att}.png'))
            plt.close()
    print('Saved plots to', args.outdir)


if __name__ == '__main__':
    main()
