
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u

# ====================================
# 1. Optical and mechanical parameters
# ====================================
OUT = Path("M31_Zemax_Stars")
OUT.mkdir(exist_ok=True)

RA0, DEC0 = 10.696, 41.289
FOV = 3.5                     # Full FOV, degrees
DELTA = 3.0                   # Outer boresight angle, degrees
MECHANICAL_R = 381.0          # Channel spacing, mm
LAMBDA_UM = 4.4
TOTAL_RAYS = 240000
PUPIL_RADIUS = 50.0           # Debug beam radius, mm
SOURCE_Z = -150.0             # Global source plane, mm
PUPIL_Z = 0.0                 # Approximate entrance plane, mm

q = np.sqrt(3) / 2 * DELTA
sky_centers = {
    "A": (0, DELTA),
    "B": (q, DELTA/2),
    "C": (q, -DELTA/2),
    "D": (0, -DELTA),
    "E": (-q, -DELTA/2),
    "F": (-q, DELTA/2),
    "G": (0, 0),
}

physical_centers = {
    name: (x * MECHANICAL_R / DELTA,
           y * MECHANICAL_R / DELTA)
    for name, (x, y) in sky_centers.items()
}

# ====================================
# 2. Local celestial coordinate basis
# +X sky: west / image right
# +Y sky: north / image up
# Ray propagation: approximately +Z
# ====================================
ra, dec = np.deg2rad([RA0, DEC0])

center_vec = np.array([
    np.cos(dec)*np.cos(ra),
    np.cos(dec)*np.sin(ra),
    np.sin(dec)
])
east = np.array([-np.sin(ra), np.cos(ra), 0])
north = np.array([
    -np.sin(dec)*np.cos(ra),
    -np.sin(dec)*np.sin(ra),
    np.cos(dec)
])

def unit(v):
    return v / np.linalg.norm(v)

def star_vector(ra_deg, dec_deg):
    a, d = np.deg2rad([ra_deg, dec_deg])
    return np.array([
        np.cos(d)*np.cos(a),
        np.cos(d)*np.sin(a),
        np.sin(d)
    ])

def boresight(x_deg, y_deg):
    return unit(
        center_vec
        - np.tan(np.deg2rad(x_deg))*east
        + np.tan(np.deg2rad(y_deg))*north
    )

boresights = {
    name: boresight(x, y)
    for name, (x, y) in sky_centers.items()
}

def angular_sep(a, b):
    return np.rad2deg(np.arccos(
        np.clip(np.dot(a, b), -1.0, 1.0)
    ))

def sky_xy(s):
    den = np.dot(s, center_vec)
    return (
        np.rad2deg(np.arctan2(-np.dot(s, east), den)),
        np.rad2deg(np.arctan2(np.dot(s, north), den))
    )

# ====================================
# 3. Query Hipparcos around M31
# ====================================
center = SkyCoord(RA0*u.deg, DEC0*u.deg, frame="icrs")

viz = Vizier(
    columns=["HIP", "RAICRS", "DEICRS", "Vmag"],
    column_filters={"Vmag": "<8.5"},
    row_limit=-1
)

tables = viz.query_region(
    center, radius=5.2*u.deg,
    catalog="I/239/hip_main"
)

if not tables:
    raise RuntimeError("No Hipparcos results. Check internet connection.")

stars = []
for row in tables[0]:
    if any(np.ma.is_masked(row[c]) for c in
           ("HIP", "RAICRS", "DEICRS", "Vmag")):
        continue

    ra_s = float(row["RAICRS"])
    dec_s = float(row["DEICRS"])
    mag = float(row["Vmag"])
    s = star_vector(ra_s, dec_s)

    visible = [
        name for name, axis in boresights.items()
        if angular_sep(s, axis) <= FOV/2
    ]
    if not visible:
        continue

    x, y = sky_xy(s)
    stars.append(dict(
        hip=int(row["HIP"]),
        ra=ra_s, dec=dec_s, mag=mag,
        vec=s, x=x, y=y,
        visible=visible
    ))

stars.sort(key=lambda a: a["mag"])
print("Catalog stars inside array:", len(stars))

# ====================================
# 4. Select 10 stars
# Force two recognizable Andromeda stars
# Then balance brightness and field coverage
# ====================================
selected = []
used = set()

def add_star(s):
    if s["hip"] not in used:
        selected.append(s)
        used.add(s["hip"])

for hip in (3881, 3231):  # Nu And, 32 And
    candidate = next((s for s in stars if s["hip"] == hip), None)
    if candidate is not None:
        add_star(candidate)

for channel in "GABCDEF":
    pool = [
        s for s in stars
        if channel in s["visible"] and s["hip"] not in used
    ]
    if pool and len(selected) < 10:
        pool.sort(key=lambda s: (
            s["mag"] - 0.4*(len(s["visible"])-1)
        ))
        add_star(pool[0])

# Prefer additional bright stars in overlap areas
remaining = [s for s in stars if s["hip"] not in used]
remaining.sort(key=lambda s: (
    -len(s["visible"]), s["mag"]
))

for s in remaining:
    if len(selected) >= 10:
        break
    # Avoid selecting almost identical star positions
    if all(angular_sep(s["vec"], t["vec"]) > 0.3
           for t in selected):
        add_star(s)

if len(selected) < 10:
    for s in stars:
        if len(selected) >= 10:
            break
        add_star(s)

# ====================================
# 5. Export star catalog
# ====================================
with open(OUT / "M31_10stars.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "HIP", "RA_deg", "Dec_deg", "Vmag",
        "SkyX_deg", "SkyY_deg", "VisibleChannels"
    ])
    for s in selected:
        writer.writerow([
            s["hip"], s["ra"], s["dec"], s["mag"],
            s["x"], s["y"], ",".join(s["visible"])
        ])

print("\nSelected stars:")
for s in selected:
    print(
        f'HIP {s["hip"]:5d} | V={s["mag"]:5.2f} | '
        f'RA={s["ra"]:.5f}, Dec={s["dec"]:.5f} | '
        f'Channels={",".join(s["visible"])}'
    )

# ====================================
# 6. Create Zemax ASCII SDF rays
# Each star is a parallel beam.
# Rays are launched toward each visible
# telescope aperture in global coordinates.
# ====================================
rng = np.random.default_rng(2026)
ray_lines = []

pairs = [(s, ch) for s in selected for ch in s["visible"]]
n_pairs = len(pairs)
rays_base = TOTAL_RAYS // n_pairs
rays_extra = TOTAL_RAYS % n_pairs
print(f"Star-channel pairs: {n_pairs}; rays per pair: {rays_base} (+1 for {rays_extra})")

for i, (s, channel) in enumerate(pairs):
    n_rays = rays_base + (1 if i < rays_extra else 0)
    v = s["vec"]

    # Propagation direction in Zemax global XYZ
    direction = unit(np.array([
        np.dot(v, east),
        -np.dot(v, north),
        np.dot(v, center_vec)
    ]))
    l, m, n = direction

    # Relative illustrative stellar flux
    # V magnitudes are NOT 4.4 um photometry
    relative_flux = 10**(-0.4*(s["mag"] - 4.5))

    pcx, pcy = physical_centers[channel]

    for _ in range(n_rays):
        r = PUPIL_RADIUS*np.sqrt(rng.random())
        phi = 2*np.pi*rng.random()

        px = pcx + r*np.cos(phi)
        py = pcy + r*np.sin(phi)

        # Propagate backward to common launch plane
        dz = PUPIL_Z - SOURCE_Z
        x0 = px - dz*l/n
        y0 = py - dz*m/n

        ray_lines.append(
            f"{x0:.7f} {y0:.7f} {SOURCE_Z:.7f} "
            f"{l:.9f} {m:.9f} {n:.9f} "
            f"{relative_flux/n_rays:.9g} "
            f"{LAMBDA_UM:.4f}\n"
        )

sdf_path = OUT / "M31_10stars.SDF"

with open(sdf_path, "w", encoding="ascii", newline="\n") as f:
    f.write(f"{len(ray_lines)} 4\n")
    f.write("! M31 region Hipparcos stars at 4.4 um\n")
    f.write("! x y z l m n intensity wavelength\n")
    f.writelines(ray_lines)

print(f"\nExported {len(ray_lines)} rays to {sdf_path}")

# ====================================
# 7. Plot sky geometry and selected stars
# ====================================
fig, ax = plt.subplots(figsize=(9, 9))

for name, (x, y) in sky_centers.items():
    circle = plt.Circle(
        (x, y), FOV/2,
        fill=False, linewidth=1.7
    )
    ax.add_patch(circle)
    ax.text(x, y, name, fontsize=13,
            ha="center", va="center")

for i, s in enumerate(selected, 1):
    ax.scatter(s["x"], s["y"], s=35)
    ax.annotate(
        f'{i}: HIP{s["hip"]}',
        (s["x"], s["y"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.set_aspect("equal")
ax.set_xlabel("Sky X (degrees; west positive)")
ax.set_ylabel("Sky Y (degrees; north positive)")
ax.set_title("M31 region: 7 FOVs and 10 catalog stars")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "M31_star_layout.png", dpi=200)
plt.show()

# ====================================
# 8. Overlay selected stars on DSS2 image
# ====================================
SKY_IMAGE = Path("cutout-CDS_P_DSS2_color.jpg")
SKY_FOV = 11.0

if SKY_IMAGE.exists():
    sky_img = plt.imread(SKY_IMAGE)
    half = SKY_FOV / 2
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(
        sky_img,
        origin="upper",
        extent=[-half, half, -half, half],
    )

    offsets = [
        (10, 10), (10, -22), (-78, 10), (10, 10), (-78, -22),
        (10, 10), (10, -22), (-78, 10), (10, 10), (-78, -22),
    ]
    for i, s in enumerate(selected, 1):
        ax2.scatter(
            s["x"], s["y"],
            s=90, facecolors="none",
            edgecolors="yellow", linewidths=1.8, zorder=5,
        )
        ax2.plot(s["x"], s["y"], "+", color="yellow",
                 markersize=8, markeredgewidth=1.2, zorder=6)
        dx, dy = offsets[i - 1]
        ax2.annotate(
            f'{i}: HIP {s["hip"]}\nV={s["mag"]:.2f}',
            (s["x"], s["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            color="yellow",
            fontsize=8,
            fontweight="bold",
            zorder=7,
        )

    ax2.set_xlim(-half, half)
    ax2.set_ylim(-half, half)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Sky X (degrees; west positive)")
    ax2.set_ylabel("Sky Y (degrees; north positive)")
    ax2.set_title("M31 DSS2 with 10 selected Hipparcos stars")
    fig2.tight_layout()
    overlay_path = OUT / "M31_stars_on_DSS2.png"
    fig2.savefig(overlay_path, dpi=200)
    plt.close(fig2)
    print(f"Saved star overlay: {overlay_path}")
else:
    print(f"Sky image not found: {SKY_IMAGE.resolve()}")
