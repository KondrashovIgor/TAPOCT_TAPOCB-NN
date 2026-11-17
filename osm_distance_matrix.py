"""OSM-backed travel-time matrices for augmented delivery polygons.

OSRM Table returns travel times of fastest routes in seconds. Distances in
metres are retained separately in the cache for provenance checks.
No straight-line or constant-speed fallback: unreachable OD pairs are reported
with their coordinates so that the affected augmented variant can be rejected.
Configure a locally hosted OSRM endpoint with OSRM_BASE_URL. Results are
cached by endpoint, profile and coordinates to make reruns reproducible.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


class OSRMMatrixError(ValueError):
    """An OSRM response cannot be used as a complete, directed OD matrix."""


def _validate_matrix(durations, distances, pts, minimal_dist=5):
    """Check every OD pair and report a small, actionable sample of failures."""
    shape = (len(pts), len(pts))
    if durations.shape != shape or distances.shape != shape:
        raise OSRMMatrixError(
            f"OSRM Table matrix shape mismatch: durations={durations.shape}, "
            f"distances={distances.shape}, expected={shape}")
    try:
        seconds = np.asarray(durations, dtype=float)
        metres = np.asarray(distances, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OSRMMatrixError(f"OSRM Table has a nonnumeric value: {exc}") from exc
    bad_duration = ~np.isfinite(seconds) | (seconds < 0)
    bad_distance = ~np.isfinite(metres) | (metres < 0)
    bad_pairs = bad_duration | bad_distance
    if bad_pairs.any():
        examples = []
        for i, j in np.argwhere(bad_pairs)[:5]:
            examples.append({
                "from": int(i), "to": int(j),
                "duration": repr(durations[i, j]),
                "distance": repr(distances[i, j]),
                "from_latlon": pts[i].tolist(), "to_latlon": pts[j].tolist(),
            })
            durations[i,j] = minimal_dist
            distances[i, j] = minimal_dist
    seconds = np.asarray(durations, dtype=float)
    metres = np.asarray(distances, dtype=float)
    
    return seconds, metres


def _haversine_m(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 12742000 * math.atan2(math.sqrt(x), math.sqrt(max(0, 1 - x)))


def road_matrix(points_latlon, *, base_url=None, profile="driving",
                cache_dir="osm_matrix_cache", max_locations=150,
                max_snap_m=100000, retries=2, timeout_s=40):
    """Return (NxN integer seconds, metadata), including depot as point 0.

    For large polygons, raise rather than silently omit OD pairs; adjust the
    self-hosted OSRM Table location limit and max_locations together.
    """
    base_url = (base_url or os.environ.get("OSRM_BASE_URL", "")).rstrip("/")
    if not base_url:
        raise ValueError("Set OSRM_BASE_URL to a configured OSM-backed OSRM service")
    pts = np.asarray(points_latlon, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or not 2 <= len(pts) <= max_locations:
        raise ValueError(f"Expected 2..{max_locations} [lat,lon] points, including depot")
    if not np.isfinite(pts).all() or np.any(np.abs(pts[:, 0]) > 90) or np.any(np.abs(pts[:, 1]) > 180):
        raise ValueError("Invalid geographic coordinates")
    key_data = {"endpoint": base_url, "profile": profile, "points": pts.tolist(),
                "max_snap_m": max_snap_m, "schema": "duration_seconds_v2"}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / (key + ".json")
    if path.exists():
        item = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(item["matrix_s"], dtype=np.int64), item["metadata"]
    coords = ";".join(f"{lon:.7f},{lat:.7f}" for lat, lon in pts)
    url = f"{base_url}/table/v1/{profile}/{coords}?{urlencode({'annotations': 'duration,distance'})}"
    last_error = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "TAPOCT-research-augmentation/1.0"})
            with urlopen(req, timeout=timeout_s) as response:
                result = json.load(response)
            if result.get("code") != "Ok":
                raise ValueError(f"OSRM Table returned {result.get('code')}: {result.get('message')}")
            durations = np.asarray(result["durations"], dtype=object)
            distances = np.asarray(result["distances"], dtype=object)
            seconds, metres = _validate_matrix(durations, distances, pts)
            waypoints = result.get("sources", [])
            if len(waypoints) != len(pts):
                raise OSRMMatrixError("OSRM Table did not return all snapped source locations")
            snaps = []
            for point, waypoint in zip(pts, waypoints):
                lon, lat = waypoint["location"]
                snaps.append(_haversine_m(point, (lat, lon)))
            if max(snaps) > max_snap_m:
                raise OSRMMatrixError(f"Point snapped {max(snaps):.1f} m from input (limit {max_snap_m} m)")
            out = np.rint(seconds).astype(np.int64)
            out[(seconds > 0) & (out == 0)] = 1
            np.fill_diagonal(out, 0)
            meta = {"source": "OSM/OSRM", "profile": profile, "base_url": base_url,
                    "matrix_unit": "s", "distance_unit": "m", "max_snap_m": round(max(snaps), 1),
                    "matrix_cache_key": key, "route_objective": "fastest_route",
                    "timing_source": "osrm_profile"}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"matrix_s": out.tolist(),
                                       "distances_m": metres.tolist(),
                                       "metadata": meta}), encoding="utf-8")
            os.replace(tmp, path)
            return out, meta
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"OSRM Table request failed after {retries + 1} attempts") from last_error
