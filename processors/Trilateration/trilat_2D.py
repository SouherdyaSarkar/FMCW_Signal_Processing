import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import fsolve

STATIONS_3D = {
    'A': np.array([0.0, 0.0, 0.0]),
    'B': np.array([8.0, 0.0, 0.0]),
    'C': np.array([4.0, 7.0, 0.0]),
}

DISTANCES = {
    'A': 6.0,
    'B': 6.5,
    'C': 5.5,
}

DEPTH_HINT = -3.0

def solve_hypocenter(stations, distances, depth_hint=-2.0):
    def equations(p):
        x, y, z = p
        keys = list(stations.keys())
        return [
            (x - stations[k][0])**2 + (y - stations[k][1])**2 + (z - stations[k][2])**2 - distances[k]**2
            for k in keys
        ]
    keys  = list(stations.keys())
    x0    = np.mean([stations[k][0] for k in keys])
    y0    = np.mean([stations[k][1] for k in keys])
    sol   = fsolve(equations, [x0, y0, depth_hint], full_output=True)[0]
    if sol[2] > 0:
        sol = fsolve(equations, [x0, y0, -depth_hint], full_output=True)[0]
    res   = {k: abs(np.sqrt(np.sum((sol - stations[k])**2)) - distances[k]) for k in keys}
    return sol, res

hypocenter, residuals = solve_hypocenter(STATIONS_3D, DISTANCES, DEPTH_HINT)

COLORS = {'A': '#4fc3f7', 'B': '#ef9a9a', 'C': '#a5d6a7'}
BG     = '#f7f9fc'
GRID   = '#dde3ec'
TEXT   = '#1a2332'
SUB    = '#5a6a7e'

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

ax2d = fig.add_subplot(gs[0])
ax3d = fig.add_subplot(gs[1], projection='3d')

ax2d.set_facecolor(BG)

theta = np.linspace(0, 2 * np.pi, 1200)

for name, pos3 in STATIONS_3D.items():
    pos2 = pos3[:2]
    r    = DISTANCES[name]
    col  = COLORS[name]
    cx   = pos2[0] + r * np.cos(theta)
    cy   = pos2[1] + r * np.sin(theta)
    ax2d.fill(cx, cy, color=col, alpha=0.08, zorder=1)
    ax2d.plot(cx, cy, color=col, linewidth=1.8, alpha=0.9, zorder=2)

epi = hypocenter[:2]

for name, pos3 in STATIONS_3D.items():
    pos2 = pos3[:2]
    col  = COLORS[name]
    ax2d.scatter(*pos2, s=90, color=col, edgecolors=TEXT, linewidths=1.2, zorder=5)
    offsets = {'A': (-0.7, -0.6), 'B': (0.2, -0.6), 'C': (-0.2, 0.4)}
    ox, oy  = offsets[name]
    ax2d.text(pos2[0] + ox, pos2[1] + oy, name, fontsize=11,
              fontweight='semibold', color=TEXT, zorder=6, ha='center')

ax2d.scatter(*epi, s=55, color=TEXT, edgecolors='white', linewidths=1.0, zorder=7)
ax2d.text(epi[0] + 0.25, epi[1] + 0.38,
          f'({epi[0]:.2f}, {epi[1]:.2f})',
          fontsize=8.5, color=SUB, zorder=8)

all_x  = [STATIONS_3D[k][0] for k in STATIONS_3D]
all_y  = [STATIONS_3D[k][1] for k in STATIONS_3D]
margin = max(DISTANCES.values()) * 1.3
xmin   = min(all_x + [epi[0]]) - margin
xmax   = max(all_x + [epi[0]]) + margin
ymin   = min(all_y + [epi[1]]) - margin
ymax   = max(all_y + [epi[1]]) + margin

ax2d.set_xlim(xmin, xmax)
ax2d.set_ylim(ymin, ymax)
ax2d.set_aspect('equal')
ax2d.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax2d.tick_params(colors=SUB, labelsize=9)
for sp in ax2d.spines.values():
    sp.set_edgecolor(GRID); sp.set_linewidth(0.8)
ax2d.set_xlabel('East  (km)', fontsize=10, color=SUB, labelpad=7)
ax2d.set_ylabel('North  (km)', fontsize=10, color=SUB, labelpad=7)
ax2d.set_title('Top View  —  Epicenter (surface projection)',
               fontsize=11, color=TEXT, pad=10, loc='left', fontweight='semibold')

legend_handles = [
    plt.Line2D([0], [0], color=COLORS[k], linewidth=2,
               label=f'Station {k}   r = {DISTANCES[k]:.1f} km')
    for k in STATIONS_3D
]
legend_handles.append(
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=TEXT,
               markersize=6, label=f'Epicenter  ({epi[0]:.2f}, {epi[1]:.2f})')
)
ax2d.legend(handles=legend_handles, loc='lower right', fontsize=9,
            framealpha=1.0, facecolor='white', edgecolor=GRID,
            labelcolor=TEXT, borderpad=0.9, handlelength=1.8)

ax3d.set_facecolor(BG)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor(GRID)
ax3d.yaxis.pane.set_edgecolor(GRID)
ax3d.zaxis.pane.set_edgecolor(GRID)
ax3d.grid(True, color=GRID, linewidth=0.5)

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 40)

for name, pos in STATIONS_3D.items():
    r   = DISTANCES[name]
    col = COLORS[name]
    sx  = pos[0] + r * np.outer(np.cos(u), np.sin(v))
    sy  = pos[1] + r * np.outer(np.sin(u), np.sin(v))
    sz  = pos[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(sx, sy, sz, color=col, alpha=0.10, linewidth=0, antialiased=True)
    for lat in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        lx = pos[0] + r * np.cos(u) * np.sin(lat)
        ly = pos[1] + r * np.sin(u) * np.sin(lat)
        lz = pos[2] + r * np.cos(lat) * np.ones_like(u)
        ax3d.plot(lx, ly, lz, color=col, linewidth=0.5, alpha=0.35)
    for lon in np.linspace(0, 2*np.pi, 8, endpoint=False):
        lx = pos[0] + r * np.cos(lon) * np.sin(v)
        ly = pos[1] + r * np.sin(lon) * np.sin(v)
        lz = pos[2] + r * np.cos(v)
        ax3d.plot(lx, ly, lz, color=col, linewidth=0.5, alpha=0.35)
    ax3d.scatter(*pos, s=70, color=col, edgecolors=TEXT, linewidths=1.1, zorder=5, depthshade=False)
    ax3d.text(pos[0], pos[1], pos[2] + 0.5, name, fontsize=10,
              fontweight='semibold', color=TEXT, ha='center')

hx, hy, hz = hypocenter

ax3d.scatter(hx, hy, hz, s=80, color=TEXT, edgecolors='white',
             linewidths=1.2, zorder=10, depthshade=False)

ax3d.plot([hx, hx], [hy, hy], [hz, 0],
          color=SUB, linewidth=1.2, linestyle='--', alpha=0.6)
ax3d.scatter(hx, hy, 0, s=40, color=SUB, marker='x', linewidths=1.5, zorder=6)

ax3d.text(hx + 0.3, hy + 0.3, hz - 0.3,
          f'({hx:.2f}, {hy:.2f}, {hz:.2f})',
          fontsize=8.5, color=SUB)

for name, pos in STATIONS_3D.items():
    col = COLORS[name]
    ax3d.plot([pos[0], hx], [pos[1], hy], [pos[2], hz],
              color=col, linewidth=1.0, linestyle=':', alpha=0.5)
    mid = (pos + hypocenter) / 2
    d   = np.linalg.norm(hypocenter - pos)
    ax3d.text(mid[0], mid[1], mid[2], f'{d:.2f} km',
              fontsize=7.5, color=col, alpha=0.85, ha='center')

z_floor = min(hz * 1.4, -1.0)
xx_f, yy_f = np.meshgrid(np.linspace(xmin, xmax, 3), np.linspace(ymin, ymax, 3))
ax3d.plot_surface(xx_f, yy_f, np.zeros_like(xx_f),
                  color='#c8d8e8', alpha=0.18, linewidth=0, zorder=0)

ax3d.set_xlabel('East  (km)', fontsize=9, color=SUB, labelpad=6)
ax3d.set_ylabel('North  (km)', fontsize=9, color=SUB, labelpad=6)
ax3d.set_zlabel('Depth  (km)', fontsize=9, color=SUB, labelpad=6)
ax3d.tick_params(colors=SUB, labelsize=8)
ax3d.set_title('3D View  —  Sphere Intersection & Hypocenter',
               fontsize=11, color=TEXT, pad=10, loc='left', fontweight='semibold')
ax3d.view_init(elev=22, azim=-55)

fig.suptitle('Seismic Trilateration', fontsize=16, fontweight='bold',
             color=TEXT, x=0.02, ha='left', y=1.01)

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/trilateration_3d.png', dpi=180,
#             bbox_inches='tight', facecolor=BG)

plt.show()
print(f"Hypocenter : ({hx:.4f}, {hy:.4f}, {hz:.4f}) km")
print(f"Depth      : {abs(hz):.4f} km")
print(f"Residuals  : { {k: f'{v:.2e}' for k,v in residuals.items()} }")