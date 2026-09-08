from __future__ import annotations

import base64
import gzip
import io
import math
import zipfile
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from shapely import contains_xy
from shapely.geometry import box
from shapely.ops import unary_union
from streamlit_folium import st_folium

# ============================================================
# BANTAY-BAHA — CLEAN REFACTOR
# ============================================================
# Real-time rainfall + 24-hour forecast driven inland flood
# screening model.
#
# WEATHER
#   Open-Meteo ECMWF IFS HRES first
#   Open-Meteo Best Match fallback
#
# TERRAIN
#   SRTM
#
# LAND / SEA
#   Natural Earth 10m Land
#
# HYDROLOGY
#   rainfall -> local runoff -> terrain concentration -> depth
#
# There is intentionally NO static "base water surface" control.
# Flood depth changes only from the selected real-time / forecast
# rainfall scenario and terrain.
#
# IMPORTANT: prototype decision-support model. It is not a validated
# hydrologic/hydraulic forecast and must be calibrated against local
# gauges and observed flood extents before operational use.
# ============================================================

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bantay-Baha",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAND_DIR = DATA_DIR / "natural_earth"
DEM_DIR = DATA_DIR / "dem"
LAND_SHP = LAND_DIR / "ne_10m_land.shp"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LAND_DIR.mkdir(parents=True, exist_ok=True)
DEM_DIR.mkdir(parents=True, exist_ok=True)

NATURAL_EARTH_LAND_URL = (
    "https://naciscdn.org/naturalearth/10m/physical/"
    "ne_10m_land.zip"
)

SRTM_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/"
    "skadi/{directory}/{tile}.hgt.gz"
)

ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"
FALLBACK_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_LAT = 13.7666
DEFAULT_LON = 122.9786
DEFAULT_GRID = 180
DEFAULT_EXTENT_KM = 25

LOCATIONS = {
    "Sipocot": (13.7666, 122.9786, 12),
    "Naga City": (13.6210, 123.1870, 11),
    "Legazpi": (13.1391, 123.7437, 11),
    "Sorsogon": (12.9722, 124.0053, 11),
    "Daet": (14.1122, 122.9553, 11),
    "Virac": (13.5848, 124.2374, 11),
    "Masbate": (12.3690, 123.6180, 11),
}

# Green -> Blue -> Yellow -> Red -> Purple -> Violet
DEPTH_STOPS = np.array(
    [0.00, 0.10, 0.30, 0.75, 1.50, 2.50, 4.00],
    dtype=np.float32,
)

DEPTH_COLORS = np.array(
    [
        [0, 180, 70, 0],
        [0, 130, 255, 210],
        [255, 235, 0, 220],
        [255, 0, 0, 225],
        [155, 0, 180, 230],
        [110, 0, 255, 235],
        [70, 0, 255, 240],
    ],
    dtype=np.float32,
)

for key, value in {
    "lat": DEFAULT_LAT,
    "lon": DEFAULT_LON,
    "zoom": 12,
    "grid_size": DEFAULT_GRID,
    "extent_km": DEFAULT_EXTENT_KM,
    "selected_date": None,
    "selected_hour": None,
}.items():
    st.session_state.setdefault(key, value)

st.session_state.setdefault("clicked_point", None)
st.session_state.setdefault("map_style", "Satellite")

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .bb-title{font-size:34px;font-weight:850;letter-spacing:-.8px;margin:0}
    .bb-subtitle{color:#64748b;font-size:13px;margin:2px 0 12px 0}
    .bb-banner{display:flex;align-items:center;justify-content:space-between;gap:12px;background:linear-gradient(135deg,#eff6ff,#f8fafc);border:1px solid #dbeafe;border-radius:12px;padding:10px 14px;margin-bottom:14px}
    .bb-banner-main{color:#0f172a;font-size:13px;font-weight:700}
    .bb-banner-sub{color:#64748b;font-size:11px;margin-top:2px}
    .bb-chip{white-space:nowrap;background:#0f172a;color:#fff;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:800}
    .point-card{background:linear-gradient(135deg,#f8fafc,#eef2ff);border:1px solid #dbe4f0;border-radius:12px;padding:13px 15px;margin-top:8px}
    .point-title{font-size:15px;font-weight:800;color:#0f172a;margin-bottom:8px}
    .point-row{display:flex;justify-content:space-between;gap:12px;padding:5px 0;border-bottom:1px solid rgba(148,163,184,.18);font-size:12px}
    .point-row:last-child{border-bottom:none}
    .point-label{color:#64748b}.point-value{color:#0f172a;font-weight:750;text-align:right}
    @media (max-width:900px){.bb-title{font-size:28px}.bb-banner{flex-direction:column;align-items:flex-start}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return default
            value = value.item()
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 1) -> str:
    return f"{safe_float(value):,.{digits}f}"


def risk_class(max_depth: float, rainfall_24h: float) -> str:
    if max_depth >= 2.50 or rainfall_24h >= 200:
        return "EXTREME"
    if max_depth >= 1.50 or rainfall_24h >= 150:
        return "SEVERE"
    if max_depth >= 0.75 or rainfall_24h >= 100:
        return "HIGH"
    if max_depth >= 0.30 or rainfall_24h >= 60:
        return "MODERATE"
    if max_depth >= 0.10 or rainfall_24h >= 30:
        return "LOW"
    if max_depth > 0:
        return "MANAGEABLE"
    return "NO FLOOD"


def nearest_time(df: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(target)
    idx = (df["time"] - target).abs().idxmin()
    return pd.Timestamp(df.loc[idx, "time"])


def nearest_weather_row(df: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    target = pd.Timestamp(target)
    idx = (df["time"] - target).abs().idxmin()
    return df.loc[idx]


def rainfall_last(df: pd.DataFrame, target: pd.Timestamp, hours: int) -> float:
    target = pd.Timestamp(target)
    start = target - pd.Timedelta(hours=hours)
    chunk = df[(df["time"] > start) & (df["time"] <= target)]
    return safe_float(chunk["precipitation"].sum())


def rainfall_next(df: pd.DataFrame, target: pd.Timestamp, hours: int) -> float:
    target = pd.Timestamp(target)
    chunk = df[df["time"] > target].head(hours)
    return safe_float(chunk["precipitation"].sum())

# ============================================================
# WEATHER
# ============================================================


def normalize_weather(weather: dict[str, Any]) -> dict[str, Any]:
    df = weather["df"].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    try:
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_localize(None)
    except Exception:
        pass
    weather["df"] = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    current_time = pd.Timestamp(weather["current_time"])
    try:
        if current_time.tzinfo is not None:
            current_time = current_time.tz_localize(None)
    except Exception:
        pass
    weather["current_time"] = current_time
    return weather


def parse_weather_payload(payload: dict[str, Any]) -> dict[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise RuntimeError("Weather API returned no hourly time series.")

    times = pd.to_datetime(
        pd.Series(hourly["time"]),
        errors="coerce",
    )
    n = len(times)

    def numeric_series(name: str) -> np.ndarray:
        values = hourly.get(name, [0.0] * n)
        return (
            pd.to_numeric(
                pd.Series(values),
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy()
        )

    df = pd.DataFrame(
        {
            "time": times,
            "precipitation": numeric_series("precipitation"),
            "rain": numeric_series("rain"),
            "temperature": numeric_series("temperature_2m"),
            "humidity": numeric_series("relative_humidity_2m"),
            "wind": numeric_series("wind_speed_10m"),
            "precipitation_probability": numeric_series(
                "precipitation_probability"
            ),
        }
    )

    df = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    current = payload.get("current", {})
    current_time = pd.to_datetime(
        current.get("time", df["time"].iloc[0])
    )

    return {
        "df": df,
        "current": current,
        "current_time": current_time,
        "timezone": payload.get("timezone", "auto"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
    }


@st.cache_data(ttl=600, show_spinner=False)
def get_weather(lat: float, lon: float) -> dict[str, Any]:
    """
    Deliberately requests only variables needed by Bantay-Baha.

    The model derives local runoff from precipitation instead of requesting a separate runoff field.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(
            [
                "precipitation",
                "rain",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ]
        ),
        "hourly": ",".join(
            [
                "precipitation",
                "rain",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation_probability",
            ]
        ),
        "past_hours": 24,
        "forecast_hours": 24,
        "timezone": "auto",
        "cell_selection": "land",
    }

    errors: list[str] = []

    for source, endpoint in (
        ("ECMWF IFS HRES 9 km", ECMWF_URL),
        ("Open-Meteo Best Match", FALLBACK_URL),
    ):
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=30,
            )

            if response.ok:
                result = parse_weather_payload(
                    response.json()
                )
                result["source"] = source
                return result

            errors.append(
                f"{source}: HTTP {response.status_code} — "
                f"{response.text[:400]}"
            )

        except requests.RequestException as exc:
            errors.append(
                f"{source}: {exc}"
            )

    raise RuntimeError(
        "Weather API failed.\n\n"
        + "\n".join(errors)
    )


# ============================================================
# NATURAL EARTH — OFFICIAL 10m LAND PRODUCT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_land_geometry():
    if not LAND_SHP.exists():
        zip_path = LAND_DIR / "ne_10m_land.zip"

        if not zip_path.exists():
            response = requests.get(
                NATURAL_EARTH_LAND_URL,
                timeout=120,
            )
            response.raise_for_status()
            zip_path.write_bytes(response.content)

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:
            archive.extractall(LAND_DIR)

    land = gpd.read_file(LAND_SHP)

    if land.crs is None:
        land = land.set_crs("EPSG:4326")
    else:
        land = land.to_crs("EPSG:4326")

    # Spatially clip the dataset to Bicol working region before union.
    region_box = box(
        122.0,
        11.5,
        125.0,
        14.7,
    )

    region_gdf = gpd.GeoDataFrame(
        geometry=[region_box],
        crs="EPSG:4326",
    )

    clipped = gpd.clip(
        land,
        region_gdf,
    )

    return unary_union(
        list(clipped.geometry)
    )


@st.cache_data(show_spinner=False)
def create_land_mask(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
) -> np.ndarray:
    geometry = get_land_geometry()
    return contains_xy(
        geometry,
        lon_grid,
        lat_grid,
    )

# ============================================================
# SRTM
# ============================================================


def srtm_tile_name(lat: float, lon: float) -> str:
    lat_i = math.floor(lat)
    lon_i = math.floor(lon)

    return (
        f"{'N' if lat_i >= 0 else 'S'}{abs(lat_i):02d}"
        f"{'E' if lon_i >= 0 else 'W'}{abs(lon_i):03d}"
    )


def srtm_directory(lat: int) -> str:
    band = (abs(lat) // 10) * 10
    return f"{'N' if lat >= 0 else 'S'}{band:02d}"


@st.cache_data(ttl=86400, show_spinner=False)
def read_srtm_tile(tile: str) -> np.ndarray:
    path = DEM_DIR / f"{tile}.hgt"

    if not path.exists():
        lat = (
            int(tile[1:3])
            if tile[0] == "N"
            else -int(tile[1:3])
        )

        response = requests.get(
            SRTM_URL.format(
                directory=srtm_directory(lat),
                tile=tile,
            ),
            timeout=120,
        )
        response.raise_for_status()
        path.write_bytes(
            gzip.decompress(response.content)
        )

    values = np.fromfile(
        path,
        dtype=">i2",
    )

    expected = 3601 * 3601

    if values.size != expected:
        raise RuntimeError(
            f"Invalid SRTM tile {tile}: {values.size} cells"
        )

    dem = values.reshape(
        3601,
        3601,
    ).astype(np.float32)

    dem[dem <= -32768] = np.nan
    return dem


@st.cache_data(show_spinner=False)
def make_grid(
    lat: float,
    lon: float,
    extent_km: float,
    grid_size: int,
):
    lat_half = extent_km / 111.0
    lon_half = extent_km / (
        111.0
        * max(
            math.cos(math.radians(lat)),
            0.2,
        )
    )

    lats = np.linspace(
        lat - lat_half,
        lat + lat_half,
        grid_size,
    )
    lons = np.linspace(
        lon - lon_half,
        lon + lon_half,
        grid_size,
    )

    lon_grid, lat_grid = np.meshgrid(
        lons,
        lats,
    )

    return lat_grid, lon_grid


@st.cache_data(ttl=86400, show_spinner=False)
def sample_dem(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> np.ndarray:
    out = np.full(
        lat_grid.shape,
        np.nan,
        dtype=np.float32,
    )

    lat_floor = np.floor(
        lat_grid
    ).astype(int)
    lon_floor = np.floor(
        lon_grid
    ).astype(int)

    tile_pairs = set(
        zip(
            lat_floor.ravel(),
            lon_floor.ravel(),
        )
    )

    for lat_i, lon_i in tile_pairs:
        tile = srtm_tile_name(
            lat_i,
            lon_i,
        )

        try:
            dem_tile = read_srtm_tile(tile)
        except Exception:
            continue

        mask = (
            (lat_floor == lat_i)
            & (lon_floor == lon_i)
        )

        if not mask.any():
            continue

        lat_values = lat_grid[mask]
        lon_values = lon_grid[mask]

        row = np.clip(
            (lat_i + 1 - lat_values) * 3600.0,
            0,
            3600,
        )

        col = np.clip(
            (lon_values - lon_i) * 3600.0,
            0,
            3600,
        )

        r0 = np.floor(row).astype(int)
        c0 = np.floor(col).astype(int)
        r1 = np.minimum(r0 + 1, 3600)
        c1 = np.minimum(c0 + 1, 3600)

        fr = row - r0
        fc = col - c0

        z00 = dem_tile[r0, c0]
        z01 = dem_tile[r0, c1]
        z10 = dem_tile[r1, c0]
        z11 = dem_tile[r1, c1]

        out[mask] = (
            z00 * (1 - fr) * (1 - fc)
            + z01 * (1 - fr) * fc
            + z10 * fr * (1 - fc)
            + z11 * fr * fc
        )

    return out


@st.cache_data(show_spinner=False)
def load_terrain(
    lat: float,
    lon: float,
    extent_km: float,
    grid_size: int,
):
    lat_grid, lon_grid = make_grid(
        lat,
        lon,
        extent_km,
        grid_size,
    )

    land_mask = create_land_mask(
        lon_grid,
        lat_grid,
    )

    dem = sample_dem(
        lat_grid,
        lon_grid,
    )

    # IMPORTANT: land only.
    dem[~land_mask] = np.nan

    valid_land = (
        land_mask
        & np.isfinite(dem)
    )

    if valid_land.any():
        fallback = float(
            np.nanmedian(
                dem[valid_land]
            )
        )

        dem[
            land_mask
            & ~np.isfinite(dem)
        ] = fallback

    # Re-apply hard ocean mask after filling.
    dem[~land_mask] = np.nan

    return (
        lat_grid,
        lon_grid,
        land_mask,
        dem,
    )

# ============================================================
# D8 FLOW CONCENTRATION
# ============================================================

@st.cache_data(show_spinner=False)
def compute_flow_index(
    dem: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    if dem.ndim != 2 or land_mask.ndim != 2:
        raise ValueError(
            f"DEM/mask must be 2-D: {dem.shape} / {land_mask.shape}"
        )

    if dem.shape != land_mask.shape:
        raise ValueError(
            f"DEM/mask shape mismatch: {dem.shape} / {land_mask.shape}"
        )

    rows, cols = dem.shape

    downstream = np.full(
        (rows, cols),
        -1,
        dtype=np.int64,
    )

    directions = (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    )

    for r in range(rows):
        for c in range(cols):
            if not land_mask[r, c]:
                continue

            z = float(dem[r, c])
            best_drop = 0.0
            best_target = -1

            for dr, dc in directions:
                rr = r + dr
                cc = c + dc

                if (
                    rr < 0
                    or rr >= rows
                    or cc < 0
                    or cc >= cols
                    or not land_mask[rr, cc]
                    or not np.isfinite(dem[rr, cc])
                ):
                    continue

                drop = z - float(dem[rr, cc])

                if drop > best_drop:
                    best_drop = drop
                    best_target = rr * cols + cc

            downstream[r, c] = best_target

    accumulation = np.zeros(
        (rows, cols),
        dtype=np.float32,
    )

    valid_flat = np.flatnonzero(
        land_mask.ravel()
    )

    accumulation.ravel()[valid_flat] = 1.0

    order = valid_flat[
        np.argsort(
            dem.ravel()[valid_flat]
        )[::-1]
    ]

    for flat in order:
        target = downstream.ravel()[flat]

        if target >= 0:
            accumulation.ravel()[target] += (
                accumulation.ravel()[flat]
            )

    values = accumulation[land_mask]

    if values.size == 0:
        return np.zeros_like(
            accumulation,
            dtype=np.float32,
        )

    p50 = float(
        np.percentile(values, 50)
    )
    p95 = float(
        np.percentile(values, 95)
    )

    denom = max(
        math.log1p(p95)
        - math.log1p(p50),
        1.0,
    )

    index = np.zeros_like(
        accumulation,
        dtype=np.float32,
    )

    index[land_mask] = np.clip(
        (
            np.log1p(accumulation[land_mask])
            - math.log1p(p50)
        )
        / denom
        * 4.0,
        0.0,
        4.0,
    )

    return index

# ============================================================
# RAINFALL-DRIVEN FLOOD MODEL
# ============================================================


def simulate_rainfall_flood(
    dem: np.ndarray,
    land_mask: np.ndarray,
    flow_index: np.ndarray,
    rainfall_24h_mm: float,
    runoff_coefficient: float,
):
    """
    Rainfall-only flood depth proxy.

    No static water-height input.

    rainfall (mm)
        -> effective runoff volume (m)
        -> terrain concentration
        -> lowland storage response
        -> depth (m)
    """

    valid_land = (
        land_mask
        & np.isfinite(dem)
    )

    depth = np.full(
        dem.shape,
        np.nan,
        dtype=np.float32,
    )

    if not valid_land.any():
        return 0.0, 0.0, depth

    rainfall = max(
        safe_float(rainfall_24h_mm),
        0.0,
    )

    coefficient = np.clip(
        safe_float(runoff_coefficient),
        0.0,
        1.0,
    )

    runoff_m = (
        rainfall
        * coefficient
        / 1000.0
    )

    elevation = dem[valid_land]

    # Low terrain receives a stronger storage response.
    low_q = float(
        np.percentile(
            elevation,
            20,
        )
    )

    elevation_scale = max(
        float(
            np.percentile(
                elevation,
                75,
            )
            - low_q
        ),
        5.0,
    )

    lowland_factor = np.zeros_like(
        dem,
        dtype=np.float32,
    )

    lowland_factor[valid_land] = np.exp(
        -np.maximum(
            dem[valid_land] - low_q,
            0.0,
        )
        / elevation_scale
    )

    concentration = (
        0.35
        + 0.95 * (flow_index / 4.0)
    )

    storage = (
        0.70
        + 1.40 * lowland_factor
    )

    depth[valid_land] = np.clip(
        runoff_m
        * concentration[valid_land]
        * storage[valid_land],
        0.0,
        4.0,
    )

    max_depth = (
        float(np.nanmax(depth))
        if np.isfinite(depth).any()
        else 0.0
    )

    return (
        runoff_m,
        max_depth,
        depth,
    )

# ============================================================
# FLOOD RASTER
# ============================================================


def create_flood_rgba(
    depth: np.ndarray,
) -> np.ndarray:
    d = np.asarray(
        depth,
        dtype=np.float32,
    )

    rgba = np.zeros(
        (*d.shape, 4),
        dtype=np.uint8,
    )

    valid = np.isfinite(d)
    flooded = valid & (d > 0)
    dry = valid & (d <= 0)

    # Dry land.
    rgba[dry, :3] = np.array(
        [35, 120, 60],
        dtype=np.uint8,
    )
    rgba[dry, 3] = 70

    if flooded.any():
        values = np.clip(
            d[flooded],
            DEPTH_STOPS[0],
            DEPTH_STOPS[-1],
        )

        idx = np.searchsorted(
            DEPTH_STOPS,
            values,
            side="right",
        ) - 1

        idx = np.clip(
            idx,
            0,
            len(DEPTH_STOPS) - 2,
        )

        lo = DEPTH_STOPS[idx]
        hi = DEPTH_STOPS[idx + 1]

        frac = (
            (values - lo)
            / np.maximum(
                hi - lo,
                1e-9,
            )
        )[:, None]

        c0 = DEPTH_COLORS[idx]
        c1 = DEPTH_COLORS[idx + 1]

        rgba[flooded] = (
            c0
            + (c1 - c0) * frac
        ).astype(np.uint8)

    # Ocean and NoData = completely transparent.
    rgba[~valid, 3] = 0

    return rgba

# ============================================================
# MAP
# ============================================================


def add_flood_legend(m: folium.Map):
    html = """
    <div style="
        position:fixed;
        left:14px;
        bottom:34px;
        z-index:9999;
        width:165px;
        background:rgba(255,255,255,.97);
        color:#111;
        padding:10px 12px;
        border:1px solid #aaa;
        border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,.25);
        font-family:Arial,sans-serif;
        font-size:11px;
        line-height:1.55;
        pointer-events:none;
    ">
      <div style="font-size:13px;font-weight:700;margin-bottom:4px;">
        Flood depth
      </div>
      <div><span style="color:#00b446">■</span> 0–0.10 m</div>
      <div><span style="color:#0082ff">■</span> 0.10–0.30 m</div>
      <div><span style="color:#d0c500">■</span> 0.30–0.75 m</div>
      <div><span style="color:#ff0000">■</span> 0.75–1.50 m</div>
      <div><span style="color:#9b00b4">■</span> 1.50–2.50 m</div>
      <div><span style="color:#6e00ff">■</span> 2.50–4.00 m</div>
      <div><span style="color:#4600ff">■</span> &gt;4.00 m</div>
    </div>
    """
    m.get_root().html.add_child(
        folium.Element(html)
    )


def make_map(
    lat: float,
    lon: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    depth: np.ndarray,
    zoom: int,
    map_style: str = "Satellite",
):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="Street",
        overlay=False,
        show=(map_style == "Street"),
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        show=(map_style == "Satellite"),
    ).add_to(m)

    lat_step = (
        abs(
            float(
                np.median(
                    np.diff(
                        lat_grid[:, 0]
                    )
                )
            )
        )
        if lat_grid.shape[0] > 1
        else 0.0
    )

    lon_step = (
        abs(
            float(
                np.median(
                    np.diff(
                        lon_grid[0, :]
                    )
                )
            )
        )
        if lon_grid.shape[1] > 1
        else 0.0
    )

    bounds = [
        [
            float(lat_grid.min()) - lat_step / 2,
            float(lon_grid.min()) - lon_step / 2,
        ],
        [
            float(lat_grid.max()) + lat_step / 2,
            float(lon_grid.max()) + lon_step / 2,
        ],
    ]

    # NumPy RGBA array, not PIL Image: avoids JSON serialization error.
    folium.raster_layers.ImageOverlay(
        image=create_flood_rgba(depth),
        bounds=bounds,
        origin="upper",
        opacity=0.82,
        interactive=False,
        cross_origin=False,
        zindex=500,
        name="Simulated Flood Depth",
    ).add_to(m)

    # Draw the actual Natural Earth land boundary.
    try:
        visible_land = get_land_geometry().intersection(
            box(
                bounds[0][1],
                bounds[0][0],
                bounds[1][1],
                bounds[1][0],
            )
        )

        if not visible_land.is_empty:
            folium.GeoJson(
                visible_land.__geo_interface__,
                name="Natural Earth Land Boundary",
                style_function=lambda _: {
                    "color": "#ffffff",
                    "weight": 1,
                    "opacity": 0.70,
                    "fillOpacity": 0,
                },
            ).add_to(m)

    except Exception:
        pass

    folium.CircleMarker(
        [lat, lon],
        radius=6,
        color="white",
        weight=2,
        fill=True,
        fill_color="#111111",
        fill_opacity=1,
        popup="Selected location",
    ).add_to(m)

    folium.LayerControl(
        collapsed=True,
        position="topright",
    ).add_to(m)

    add_flood_legend(m)

    return m

# ============================================================
# MAIN
# ============================================================


def main():
    st.markdown(
        '<div class="bb-title">🌊 Bantay-Baha</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="bb-subtitle">'
        'Real-time rainfall-driven inland flood simulation '
        'for the next 24 hours'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------
    with st.sidebar:
        st.header("📡 Real-Time Control")
        st.caption("Choose a location and explore the next 24 hours.")

        location_name = st.selectbox(
            "Location",
            list(LOCATIONS.keys()),
        )

        st.session_state.map_style = st.radio(
            "Base map",
            ["Satellite", "Street"],
            index=0 if st.session_state.map_style == "Satellite" else 1,
            horizontal=True,
        )

        preset_lat, preset_lon, preset_zoom = LOCATIONS[
            location_name
        ]

        if st.button(
            "📍 Use selected location",
            use_container_width=True,
        ):
            st.session_state.lat = preset_lat
            st.session_state.lon = preset_lon
            st.session_state.zoom = preset_zoom
            load_terrain.clear()
            create_land_mask.clear()
            compute_flow_index.clear()
            get_weather.clear()
            st.rerun()

        st.session_state.lat = st.number_input(
            "Latitude",
            value=float(st.session_state.lat),
            format="%.6f",
        )

        st.session_state.lon = st.number_input(
            "Longitude",
            value=float(st.session_state.lon),
            format="%.6f",
        )

        extent_km = st.slider(
            "Simulation radius",
            min_value=5,
            max_value=50,
            value=int(st.session_state.extent_km),
            step=5,
            format="%d km",
        )

        grid_size = st.select_slider(
            "Map resolution",
            options=[120, 150, 180, 210],
            value=int(st.session_state.grid_size),
        )

        st.divider()
        st.subheader("🌧️ Runoff")

        runoff_coefficient = st.slider(
            "Runoff coefficient",
            min_value=0.10,
            max_value=1.00,
            value=0.65,
            step=0.05,
        )

        st.caption(
            "No static water-height simulation. "
            "Flooding responds to rainfall accumulated through the selected time."
        )

        if st.button(
            "🔄 Refresh weather",
            use_container_width=True,
        ):
            get_weather.clear()
            st.rerun()

        if st.button(
            "🧹 Clear caches",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------
    try:
        weather = normalize_weather(
            get_weather(
                st.session_state.lat,
                st.session_state.lon,
            )
        )
    except Exception as exc:
        st.error("Weather service error")
        st.code(str(exc))
        st.stop()

    weather_df = weather["df"]
    current_time = weather["current_time"]

    # --------------------------------------------------------
    # Date / horizon
    # --------------------------------------------------------
    st.subheader("📅 Real-Time Simulation")

    forecast_end = current_time + pd.Timedelta(hours=24)

    timeline = weather_df[
        (weather_df["time"] >= current_time)
        & (weather_df["time"] <= forecast_end)
    ].copy()

    if timeline.empty:
        st.error("No current-to-24-hour forecast timestamps returned.")
        st.stop()

    horizon_options = list(range(len(timeline)))

    horizon = st.slider(
        "Simulation horizon",
        min_value=0,
        max_value=min(24, len(horizon_options) - 1),
        value=0,
        step=1,
        format="+%d h",
    )

    selected_time = pd.Timestamp(
        timeline.iloc[horizon]["time"]
    )

    if horizon == 0:
        mode = "🟢 CURRENT"
    else:
        mode = f"🔵 +{horizon}h FORECAST"

    st.caption(
        f"{mode} • {selected_time.strftime('%A, %d %B %Y • %H:%M')} • "
        f"{weather['source']}"
    )

    # --------------------------------------------------------
    # Weather metrics
    # --------------------------------------------------------
    rain1 = rainfall_last(
        weather_df,
        selected_time,
        1,
    )
    rain6 = rainfall_last(
        weather_df,
        selected_time,
        6,
    )
    rain12 = rainfall_last(
        weather_df,
        selected_time,
        12,
    )
    rain24 = rainfall_last(
        weather_df,
        selected_time,
        24,
    )
    next6 = rainfall_next(
        weather_df,
        selected_time,
        6,
    )
    next24 = rainfall_next(
        weather_df,
        selected_time,
        24,
    )

    # The ECMWF / Open-Meteo `current` object can omit some derived
    # fields depending on the selected model/grid. The hourly series is
    # therefore the authoritative fallback for the live card.
    live_row = nearest_weather_row(
        weather_df,
        current_time,
    )

    row = nearest_weather_row(
        weather_df,
        selected_time,
    )

    def live_value(api_key: str, hourly_column: str) -> float:
        api_value = weather["current"].get(api_key)
        value = safe_float(api_value, float("nan"))
        if not np.isfinite(value):
            value = safe_float(live_row.get(hourly_column), 0.0)
        return value

    current_rain = live_value(
        "precipitation",
        "precipitation",
    )

    current_temp = live_value(
        "temperature_2m",
        "temperature",
    )

    current_humidity = live_value(
        "relative_humidity_2m",
        "humidity",
    )

    current_wind = live_value(
        "wind_speed_10m",
        "wind",
    )

    scores = st.columns(8)

    score_values = [
        ("Current rain", f"{fmt(current_rain)} mm/h"),
        ("1h rain", f"{fmt(rain1)} mm"),
        ("6h rain", f"{fmt(rain6)} mm"),
        ("12h rain", f"{fmt(rain12)} mm"),
        ("24h rain", f"{fmt(rain24)} mm"),
        ("Next 6h", f"{fmt(next6)} mm"),
        ("Next 24h", f"{fmt(next24)} mm"),
        ("Temperature", f"{fmt(current_temp)} °C"),
    ]

    for column, (label, value) in zip(
        scores,
        score_values,
    ):
        with column:
            st.metric(
                label,
                value,
            )

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------
    with st.spinner(
        "Loading Natural Earth land boundary + SRTM terrain…"
    ):
        lat_grid, lon_grid, land_mask, dem = load_terrain(
            st.session_state.lat,
            st.session_state.lon,
            extent_km,
            grid_size,
        )

    with st.spinner(
        "Building D8 terrain flow network…"
    ):
        flow_index = compute_flow_index(
            dem,
            land_mask,
        )

    # --------------------------------------------------------
    # Rainfall-driven flood
    # --------------------------------------------------------
    runoff_m, max_depth, depth = simulate_rainfall_flood(
        dem,
        land_mask,
        flow_index,
        rain24,
        runoff_coefficient,
    )

    valid_land = (
        land_mask
        & np.isfinite(dem)
    )

    valid_depth = (
        valid_land
        & np.isfinite(depth)
    )

    flooded = (
        valid_depth
        & (depth >= 0.10)
    )

    land_cells = int(
        valid_land.sum()
    )

    flood_cells = int(
        flooded.sum()
    )

    affected_percent = (
        100.0 * flood_cells / land_cells
        if land_cells
        else 0.0
    )

    mean_depth = (
        float(np.nanmean(depth[flooded]))
        if flooded.any()
        else 0.0
    )

    risk = risk_class(
        max_depth,
        rain24,
    )

    # --------------------------------------------------------
    # Main layout
    # --------------------------------------------------------
    left, center, right = st.columns(
        [0.95, 2.50, 1.05]
    )

    with left:
        st.subheader("🌊 Flood")

        st.metric(
            "Risk",
            risk,
        )

        st.metric(
            "Max depth",
            f"{fmt(max_depth, 2)} m",
        )

        st.metric(
            "Mean flooded depth",
            f"{fmt(mean_depth, 2)} m",
        )

        st.metric(
            "Land affected",
            f"{fmt(affected_percent, 1)}%",
        )

        st.divider()

        st.subheader("💧 Runoff")
        st.metric(
            "Effective runoff",
            f"{fmt(runoff_m * 1000, 1)} mm",
        )

        st.caption(
            "Rainfall → runoff → terrain concentration → simulated depth"
        )

    with center:
        st.subheader(
            f"🗺️ {location_name} • Flood Simulation"
        )
        st.caption(
            f"Scenario: {selected_time.strftime('%d %B %Y • %H:%M')} • {mode}"
        )

        fmap = make_map(
            st.session_state.lat,
            st.session_state.lon,
            lat_grid,
            lon_grid,
            depth,
            int(st.session_state.zoom),
            st.session_state.map_style,
        )

        map_result = st_folium(
            fmap,
            width=None,
            height=625,
            returned_objects=["last_clicked"],
            key="bantay_baha_flood_map",
        )

        if map_result and map_result.get("last_clicked"):
            st.session_state.clicked_point = {
                "lat": float(map_result["last_clicked"]["lat"]),
                "lon": float(map_result["last_clicked"]["lng"]),
            }

        st.caption(
            "Click anywhere on the map to inspect the selected point. "
            "Ocean cells are hard-masked by Natural Earth."
        )

        point = st.session_state.clicked_point

        if point is None:
            st.info("📍 Click the map to open Point Details.")
        else:
            click_lat = point["lat"]
            click_lon = point["lon"]

            distance2 = (
                (lat_grid - click_lat) ** 2
                + (lon_grid - click_lon) ** 2
            )

            point_row, point_col = np.unravel_index(
                int(np.nanargmin(distance2)),
                distance2.shape,
            )

            on_land = bool(
                land_mask[point_row, point_col]
                and np.isfinite(dem[point_row, point_col])
            )

            elevation_value = (
                float(dem[point_row, point_col])
                if on_land
                else None
            )

            depth_value = (
                float(depth[point_row, point_col])
                if on_land and np.isfinite(depth[point_row, point_col])
                else 0.0
            )

            if not on_land:
                surface_label = "🌊 OCEAN / OUTSIDE LAND MASK"
                point_risk = "N/A"
            elif depth_value >= 1.50:
                surface_label = "🟣 SEVERE FLOOD"
                point_risk = "SEVERE"
            elif depth_value >= 0.75:
                surface_label = "🔴 DANGEROUS FLOOD"
                point_risk = "HIGH"
            elif depth_value >= 0.30:
                surface_label = "🟡 FLOODED"
                point_risk = "MODERATE"
            elif depth_value >= 0.10:
                surface_label = "🔵 MINOR FLOODING"
                point_risk = "LOW"
            else:
                surface_label = "🟢 DRY LAND"
                point_risk = "NO FLOOD"

            point_weather = nearest_weather_row(weather_df, selected_time)

            st.markdown(
                f"""
                <div class="point-card">
                  <div class="point-title">📍 Point Details</div>
                  <div class="point-row"><span class="point-label">Coordinates</span><span class="point-value">{click_lat:.5f}, {click_lon:.5f}</span></div>
                  <div class="point-row"><span class="point-label">Surface</span><span class="point-value">{surface_label}</span></div>
                  <div class="point-row"><span class="point-label">Elevation</span><span class="point-value">{f"{elevation_value:.2f} m" if elevation_value is not None else "NoData"}</span></div>
                  <div class="point-row"><span class="point-label">Flood depth</span><span class="point-value">{depth_value:.2f} m</span></div>
                  <div class="point-row"><span class="point-label">Rain at scenario time</span><span class="point-value">{safe_float(point_weather['precipitation']):.1f} mm/h</span></div>
                  <div class="point-row"><span class="point-label">24h rainfall</span><span class="point-value">{rain24:.1f} mm</span></div>
                  <div class="point-row"><span class="point-label">Simulation</span><span class="point-value">+{horizon} h • {selected_time.strftime('%d %b %H:%M')}</span></div>
                  <div class="point-row"><span class="point-label">Point risk</span><span class="point-value">{point_risk}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("✕ Clear point", use_container_width=True):
                st.session_state.clicked_point = None
                st.rerun()

    with right:
        st.subheader("📡 Current")
        st.caption(
            f"Live model • {current_time.strftime('%d %b %H:%M')} • {weather['source']}"
        )

        st.metric(
            "Rain",
            f"{fmt(current_rain)} mm/h",
        )
        st.metric(
            "Temperature",
            f"{fmt(current_temp)} °C",
        )
        st.metric(
            "Humidity",
            f"{fmt(current_humidity)}%",
        )
        st.metric(
            "Wind",
            f"{fmt(current_wind)} km/h",
        )

        st.divider()

        st.subheader("🔮 Next 24h")
        st.metric(
            "Rain",
            f"{fmt(next24)} mm",
        )
        st.metric(
            "Projected max depth",
            f"{fmt(max_depth, 2)} m",
        )
        st.metric(
            "Projected risk",
            risk,
        )

    # --------------------------------------------------------
    # Rain chart
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("🌧️ Real-Time + 24-Hour Rainfall")

    chart_df = weather_df[
        (
            weather_df["time"]
            >= current_time - pd.Timedelta(hours=6)
        )
        & (
            weather_df["time"]
            <= current_time + pd.Timedelta(hours=24)
        )
    ].copy()

    fig = go.Figure()

    recent = chart_df[
        chart_df["time"] <= current_time
    ]

    forecast = chart_df[
        chart_df["time"] > current_time
    ]

    fig.add_trace(
        go.Bar(
            x=recent["time"],
            y=recent["precipitation"],
            name="Recent",
        )
    )

    fig.add_trace(
        go.Bar(
            x=forecast["time"],
            y=forecast["precipitation"],
            name="Forecast",
        )
    )

    fig.add_vline(
        x=selected_time,
        line_dash="dash",
        annotation_text="Simulation",
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        yaxis_title="Rainfall (mm/h)",
        xaxis_title="Time",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 24h flood outlook
    # --------------------------------------------------------
    st.subheader("🔮 24-Hour Flood Outlook")

    outlook_rows = []

    for index, future_time in enumerate(
        timeline["time"].tolist()
    ):
        future_time = pd.Timestamp(
            future_time
        )

        rain24_future = rainfall_last(
            weather_df,
            future_time,
            24,
        )

        _, future_max_depth, _ = simulate_rainfall_flood(
            dem,
            land_mask,
            flow_index,
            rain24_future,
            runoff_coefficient,
        )

        outlook_rows.append(
            {
                "Hour": index,
                "Time": future_time.strftime(
                    "%d %b %H:%M"
                ),
                "24h rain (mm)": round(
                    rain24_future,
                    1,
                ),
                "Max depth (m)": round(
                    future_max_depth,
                    2,
                ),
                "Risk": risk_class(
                    future_max_depth,
                    rain24_future,
                ),
            }
        )

    outlook_df = pd.DataFrame(
        outlook_rows
    )

    st.dataframe(
        outlook_df,
        use_container_width=True,
        hide_index=True,
        height=330,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    with st.expander("🔎 Diagnostics"):
        valid_elev = dem[
            valid_land
        ]

        st.json(
            {
                "weather_source": weather["source"],
                "weather_timezone": weather["timezone"],
                "weather_latitude": safe_float(
                    weather["latitude"]
                ),
                "weather_longitude": safe_float(
                    weather["longitude"]
                ),
                "weather_elevation_m": safe_float(
                    weather["elevation"]
                ),
                "current_time": str(
                    current_time
                ),
                "selected_time": str(
                    selected_time
                ),
                "horizon_hours": horizon,
                "grid": list(dem.shape),
                "land_cells": land_cells,
                "flood_cells_0_10m": flood_cells,
                "dem_min_m": (
                    round(
                        float(np.min(valid_elev)),
                        2,
                    )
                    if valid_elev.size
                    else None
                ),
                "dem_max_m": (
                    round(
                        float(np.max(valid_elev)),
                        2,
                    )
                    if valid_elev.size
                    else None
                ),
                "rain_24h_mm": round(
                    rain24,
                    2,
                ),
                "next_24h_mm": round(
                    next24,
                    2,
                ),
                "effective_runoff_mm": round(
                    runoff_m * 1000,
                    2,
                ),
                "max_depth_m": round(
                    max_depth,
                    3,
                ),
                "risk": risk,
            }
        )

    st.caption(
        "Weather: Open-Meteo ECMWF IFS HRES • "
        "Land: Natural Earth 10m • Terrain: SRTM • "
        "Routing: D8"
    )

    st.warning(
        "Prototype only: flood depth is a rainfall-driven screening proxy. "
        "Validate against PAGASA/LGU gauges, observed flood extents, soil, "
        "land cover, drainage and validated hydrologic/hydraulic models "
        "before operational use."
    )


if __name__ == "__main__":
    main()
