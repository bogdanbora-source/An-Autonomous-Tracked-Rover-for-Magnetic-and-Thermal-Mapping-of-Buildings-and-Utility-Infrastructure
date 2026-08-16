#!/usr/bin/env python3
"""
============================================================
 MAGNETIC FIELD MAPPING — post-processing for the rover
============================================================

Reads the CSV files the rover writes to the SD card:

  date,time,lat,lon,mag_x,mag_y,mag_z,mag_total,heading_deg,front_dist_cm

(an optional 'alt_m' column after 'lon' is auto-detected if you
ever add altitude logging to the firmware)

OUTPUTS
-------
1. rover_map.html          interactive map (open in any browser):
                             - driven track per floor
                             - color-coded anomaly points per floor
                             - magnetic intensity heat layer per floor
                             - obstacle encounter markers
                             - toggleable layers (one per floor)
2. floor<N>_heatmap.png    interpolated magnetic heatmap per floor
3. summary printed to the terminal

MULTI-FLOOR / APARTMENT BLOCK WORKFLOW
--------------------------------------
GPS altitude cannot distinguish building floors (10-20 m error vs
~3 m per floor), so floors are assigned PER FILE: do one rover run
per floor (each power-cycle creates a new LOGxxx.CSV), then:

  python rover_map.py LOG000.CSV LOG001.CSV LOG002.CSV --floors 0 1 2

Single outdoor run:

  python rover_map.py LOG000.CSV

If --floors is omitted, all files are treated as floor 0 (merged).

Requires:  pip install pandas numpy scipy matplotlib folium
============================================================
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import folium
from folium.plugins import HeatMap
import branca.colormap as bcm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------
OBSTACLE_CM = 30          # rows with front_dist below this = obstacle event
GRID_RES = 120            # interpolation grid resolution for PNG heatmaps
MIN_POINTS_FOR_HEATMAP = 8
FLOOR_HEIGHT_M = 3.0      # vertical spacing between floors in the 3D view


# ------------------------------------------------------------
#  LOADING & CLEANING
# ------------------------------------------------------------
def load_log(path: Path, floor: int) -> pd.DataFrame:
    """Load one rover CSV, clean it, tag it with its floor."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"lat", "lon", "mag_x", "mag_y", "mag_z", "mag_total"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")

    total_rows = len(df)

    # numeric coercion (bad/partial rows -> NaN)
    num_cols = [c for c in df.columns if c not in ("date", "time")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # timestamp (optional — GPS may not have had a fix yet)
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce", utc=True,
        )

    # split: rows with a GPS fix are mappable, others only count for stats
    has_fix = df["lat"].notna() & df["lon"].notna() & (df["lat"] != 0) & (df["lon"] != 0)
    no_fix = int((~has_fix).sum())
    df = df[has_fix].copy()

    # drop physically impossible magnetometer glitches (I2C hiccups)
    df = df[df["mag_total"].between(1, 1e6)]

    df["floor"] = floor
    df["source_file"] = path.name

    print(f"  {path.name}: {total_rows} rows, {no_fix} without GPS fix "
          f"(skipped), {len(df)} mappable -> floor {floor}")
    return df


def check_calibration(df: pd.DataFrame):
    """
    Earth's field magnitude is essentially CONSTANT at one location.
    A well-calibrated magnetometer therefore reports a near-constant
    |B| no matter which way the rover is pointing — only X/Y/Z shift.

    If |B| tracks HEADING, the sensor is miscalibrated (or was sampled
    while the motors were running), and every turn the rover makes will
    look like a magnetic anomaly. Mapping that data produces a map of
    the driving, not of the ground. So we check, loudly.
    """
    if "heading_deg" not in df.columns or df["heading_deg"].isna().all():
        return

    b = df["mag_total"]
    spread = b.max() / max(b.min(), 1e-6)

    # bin |B| by heading and see how much the mean moves
    bins = pd.cut(df["heading_deg"], bins=np.arange(-180, 181, 45))
    means = df.groupby(bins, observed=True)["mag_total"].mean().dropna()
    heading_swing = (means.max() / max(means.min(), 1e-6)) if len(means) >= 3 else 1.0

    print("\n--- calibration check ---")
    print(f"  |B| overall spread : {spread:.1f}x")
    print(f"  |B| variation with heading: {heading_swing:.1f}x")

    if heading_swing >= 1.8:
        print("  *** WARNING: |B| depends strongly on heading. ***")
        print("  The magnetometer is miscalibrated, or was sampled while the")
        print("  motors were running. Turns will masquerade as anomalies and")
        print("  this map is NOT trustworthy. Recalibrate (tilt through all")
        print("  orientations) and use stop-and-sample logging.")
    elif heading_swing >= 1.35:
        print("  Caution: some heading dependence remains. Weak anomalies")
        print("  may be unreliable; strong ones are probably still real.")
    else:
        print("  OK — |B| is largely heading-independent. Data looks trustworthy.")

    if "was_moving" in df.columns and df["was_moving"].notna().any():
        moving = int((df["was_moving"] == 1).sum())
        if moving:
            print(f"  note: {moving} rows were logged WHILE MOVING "
                  f"(was_moving=1) — those carry motor noise.")


def compute_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Magnetic anomaly = deviation of mag_total from that floor's median.
    The median (not mean) is the baseline: robust against the big spikes
    we are actually hunting for. Positive = stronger than ambient field,
    negative = weaker. Units are raw QMC5883L counts (relative), which is
    exactly what you want for anomaly hunting — absolute uT calibration
    is unnecessary and the cheap module isn't lab-calibrated anyway.
    """
    df = df.copy()
    df["anomaly"] = np.nan
    for fl, idx in df.groupby("floor").groups.items():
        base = df.loc[idx, "mag_total"].median()
        df.loc[idx, "anomaly"] = df.loc[idx, "mag_total"] - base
    return df


# ------------------------------------------------------------
#  LOCAL METRIC PROJECTION (for interpolation in meters)
# ------------------------------------------------------------
def to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular projection around the survey centre — accurate to
    millimetres at rover-survey scale (hundreds of meters)."""
    x = (np.asarray(lon) - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * 110_540.0
    return x, y


# ------------------------------------------------------------
#  INTERACTIVE FOLIUM MAP
# ------------------------------------------------------------
def build_folium_map(df: pd.DataFrame, out_html: Path):
    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=19, max_zoom=22,
                   tiles="OpenStreetMap", control_scale=True)
    # satellite layer option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", max_zoom=22,
    ).add_to(m)

    # one diverging colour scale shared by all floors -> comparable colours
    amax = max(abs(df["anomaly"].quantile(0.02)),
               abs(df["anomaly"].quantile(0.98)), 1.0)
    cmap = bcm.LinearColormap(
        ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        vmin=-amax, vmax=amax,
        caption="Magnetic anomaly (raw counts vs floor median)",
    )
    cmap.add_to(m)

    for fl in sorted(df["floor"].unique()):
        sub = df[df["floor"] == fl].sort_values("timestamp", na_position="last")

        fg_track = folium.FeatureGroup(name=f"Floor {fl} — track", show=(fl == sorted(df['floor'].unique())[0]))
        fg_pts   = folium.FeatureGroup(name=f"Floor {fl} — anomaly points", show=(fl == sorted(df['floor'].unique())[0]))
        fg_heat  = folium.FeatureGroup(name=f"Floor {fl} — intensity heat", show=False)
        fg_obs   = folium.FeatureGroup(name=f"Floor {fl} — obstacles", show=False)

        # --- driven track ---
        coords = sub[["lat", "lon"]].values.tolist()
        if len(coords) >= 2:
            folium.PolyLine(coords, weight=2, opacity=0.6,
                            color="#555555").add_to(fg_track)

        # --- anomaly points ---
        for _, r in sub.iterrows():
            popup = (f"<b>Floor {fl}</b><br>"
                     f"time: {r.get('timestamp', '')}<br>"
                     f"|B|: {r['mag_total']:.0f}  "
                     f"(anomaly {r['anomaly']:+.0f})<br>"
                     f"X/Y/Z: {r['mag_x']:.0f}/{r['mag_y']:.0f}/{r['mag_z']:.0f}<br>"
                     f"heading: {r.get('heading_deg', float('nan')):.0f} deg")
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5,
                color=None, fill=True, fill_opacity=0.85,
                fill_color=cmap(float(np.clip(r["anomaly"], -amax, amax))),
                popup=folium.Popup(popup, max_width=250),
            ).add_to(fg_pts)

        # --- intensity heat layer ---
        lo, hi = sub["mag_total"].min(), sub["mag_total"].max()
        span = (hi - lo) or 1.0
        heat_data = [[r["lat"], r["lon"], (r["mag_total"] - lo) / span]
                     for _, r in sub.iterrows()]
        if heat_data:
            HeatMap(heat_data, radius=18, blur=22, max_zoom=22).add_to(fg_heat)

        # --- obstacle encounters ---
        if "front_dist_cm" in sub.columns:
            for _, r in sub[sub["front_dist_cm"] < OBSTACLE_CM].iterrows():
                folium.Marker(
                    [r["lat"], r["lon"]],
                    icon=folium.Icon(color="black", icon="ban-circle"),
                    popup=f"Obstacle @ {r['front_dist_cm']:.0f} cm (floor {fl})",
                ).add_to(fg_obs)

        for fg in (fg_track, fg_pts, fg_heat, fg_obs):
            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))
    print(f"\nInteractive map -> {out_html}")


# ------------------------------------------------------------
#  INTERPOLATED PNG HEATMAP PER FLOOR
# ------------------------------------------------------------
def build_png_heatmaps(df: pd.DataFrame, out_dir: Path):
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()

    for fl in sorted(df["floor"].unique()):
        sub = df[df["floor"] == fl]
        if len(sub) < MIN_POINTS_FOR_HEATMAP:
            print(f"  floor {fl}: only {len(sub)} points, skipping PNG "
                  f"(need {MIN_POINTS_FOR_HEATMAP})")
            continue

        x, y = to_local_xy(sub["lat"], sub["lon"], lat0, lon0)
        z = sub["mag_total"].values

        xi = np.linspace(x.min(), x.max(), GRID_RES)
        yi = np.linspace(y.min(), y.max(), GRID_RES)
        XI, YI = np.meshgrid(xi, yi)

        # linear interpolation inside the surveyed area; 'nearest' fallback
        # fills the hull edges so the plot has no ragged NaN border
        ZI = griddata((x, y), z, (XI, YI), method="linear")
        ZN = griddata((x, y), z, (XI, YI), method="nearest")
        ZI = np.where(np.isnan(ZI), ZN, ZI)

        fig, ax = plt.subplots(figsize=(10, 8))
        pc = ax.pcolormesh(XI, YI, ZI, shading="auto", cmap="RdYlBu_r")
        ax.plot(x, y, ".", ms=3, color="black", alpha=0.4, label="samples")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title(f"Magnetic field intensity — floor {fl}\n"
                     f"(raw QMC5883L counts, {len(sub)} samples)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        fig.colorbar(pc, ax=ax, label="|B| (raw counts)")
        out = out_dir / f"floor{fl}_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  floor {fl}: heatmap -> {out}")


# ------------------------------------------------------------
#  3D "BLOCK" VISUALISATION (multi-floor)
# ------------------------------------------------------------
def build_3d_block(df: pd.DataFrame, out_html: Path, floor_h: float):
    """
    Stack each floor's interpolated magnetic heatmap as a horizontal
    surface at its real height (floor * floor_h meters). Interactive
    HTML: rotate, zoom, toggle floors in the legend. Anomalies that
    line up vertically across floors = risers/shafts/through-pipes.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (3D view skipped — install with: pip install plotly)")
        return

    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    floors = sorted(df["floor"].unique())

    # shared colour range so the same colour means the same field
    # strength on every floor
    cmin = df["mag_total"].quantile(0.02)
    cmax = df["mag_total"].quantile(0.98)

    fig = go.Figure()

    for fl in floors:
        sub = df[df["floor"] == fl]
        if len(sub) < MIN_POINTS_FOR_HEATMAP:
            print(f"  floor {fl}: too few points for 3D surface, "
                  f"plotting samples only")
        z_level = fl * floor_h

        x, y = to_local_xy(sub["lat"], sub["lon"], lat0, lon0)

        if len(sub) >= MIN_POINTS_FOR_HEATMAP:
            xi = np.linspace(x.min(), x.max(), GRID_RES)
            yi = np.linspace(y.min(), y.max(), GRID_RES)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((x, y), sub["mag_total"].values, (XI, YI),
                          method="linear")
            ZN = griddata((x, y), sub["mag_total"].values, (XI, YI),
                          method="nearest")
            ZI = np.where(np.isnan(ZI), ZN, ZI)

            fig.add_trace(go.Surface(
                x=XI, y=YI,
                z=np.full_like(ZI, z_level),
                surfacecolor=ZI,
                colorscale="RdYlBu_r",
                cmin=cmin, cmax=cmax,
                opacity=0.92,
                name=f"Floor {fl}",
                showlegend=True,
                colorbar=dict(title="|B| (raw counts)", len=0.6),
                showscale=bool(fl == floors[0]),
                hovertemplate=("E %{x:.1f} m, N %{y:.1f} m"
                               f"<br>floor {fl}"
                               "<br>|B| %{surfacecolor:.0f}"
                               "<extra></extra>"),
            ))

        # driven track slightly above the surface
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=np.full(len(sub), z_level + 0.25),
            mode="lines",
            line=dict(color="rgba(40,40,40,0.55)", width=2),
            name=f"Floor {fl} track",
            showlegend=True,
        ))

        # strongest anomaly per floor, flagged
        peak = sub.loc[sub["anomaly"].abs().idxmax()]
        px_, py_ = to_local_xy(peak["lat"], peak["lon"], lat0, lon0)
        fig.add_trace(go.Scatter3d(
            x=[float(px_)], y=[float(py_)], z=[z_level + 0.6],
            mode="markers+text",
            marker=dict(size=5, color="black", symbol="diamond"),
            text=[f"{peak['anomaly']:+.0f}"],
            textposition="top center",
            name=f"Floor {fl} peak anomaly",
            showlegend=False,
        ))

    fig.update_layout(
        title="Magnetic survey — building view "
              f"({len(floors)} floors, {floor_h:.0f} m spacing)",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Height (m)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(str(out_html), include_plotlyjs=True)
    print(f"3D building view -> {out_html}")


# ------------------------------------------------------------
#  SUMMARY
# ------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n===== SURVEY SUMMARY =====")
    for fl in sorted(df["floor"].unique()):
        sub = df[df["floor"] == fl]
        dur = ""
        if "timestamp" in sub.columns and sub["timestamp"].notna().any():
            t0, t1 = sub["timestamp"].min(), sub["timestamp"].max()
            dur = f", {t0:%H:%M:%S}-{t1:%H:%M:%S} UTC"
        strongest = sub.loc[sub["anomaly"].abs().idxmax()]
        print(f"Floor {fl}: {len(sub)} points{dur}")
        print(f"   |B| median {sub['mag_total'].median():.0f}, "
              f"range {sub['mag_total'].min():.0f}..{sub['mag_total'].max():.0f}")
        print(f"   strongest anomaly {strongest['anomaly']:+.0f} at "
              f"({strongest['lat']:.6f}, {strongest['lon']:.6f})")
        if "front_dist_cm" in sub.columns:
            print(f"   obstacle events: {(sub['front_dist_cm'] < OBSTACLE_CM).sum()}")


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build magnetic field maps from rover SD-card CSV logs.")
    ap.add_argument("files", nargs="+", help="LOGxxx.CSV files from the SD card")
    ap.add_argument("--floors", nargs="*", type=int, default=None,
                    help="floor number for each file, same order "
                         "(default: all floor 0)")
    ap.add_argument("--out", default="rover_map_output",
                    help="output directory (default: rover_map_output)")
    ap.add_argument("--floor-height", type=float, default=FLOOR_HEIGHT_M,
                    help="vertical spacing between floors in the 3D view, "
                         f"meters (default {FLOOR_HEIGHT_M})")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            sys.exit(f"ERROR: file not found: {f}")

    if args.floors is None:
        floors = [0] * len(files)
    elif len(args.floors) != len(files):
        sys.exit("ERROR: --floors must list one floor per file")
    else:
        floors = args.floors

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading logs:")
    frames = [load_log(f, fl) for f, fl in zip(files, floors)]
    df = pd.concat(frames, ignore_index=True)

    if df.empty:
        sys.exit("\nERROR: no rows with a GPS fix in any file — nothing to map.\n"
                 "(Indoor runs without GPS fix cannot be positioned.)")

    check_calibration(df)
    df = compute_anomaly(df)

    build_folium_map(df, out_dir / "rover_map.html")
    build_png_heatmaps(df, out_dir)
    if df["floor"].nunique() > 1:
        build_3d_block(df, out_dir / "rover_map_3d.html", args.floor_height)
    print_summary(df)
    print("\nDone. Open rover_map.html in a browser and toggle layers.")


if __name__ == "__main__":
    main()
