"""Fetch the administrative Pohjois-Savo polygon and its unsimplified drive graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import osmnx as ox
import requests
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "pohjois-savo-dijkstra/1.0 (educational one-off OSM road-network download)"
GRAPH_OPTIONS = dict(network_type="drive", simplify=False, retain_all=True, truncate_by_edge=False)
ATTRIBUTION = "© OpenStreetMap contributors; https://www.openstreetmap.org/copyright"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_boundary(result: dict):
    tags = result.get("extratags", {})
    names = result.get("namedetails", {})
    region_code = tags.get("ISO3166-2", names.get("ISO3166-2"))
    if (result.get("osm_type") != "relation"
            or result.get("category", result.get("class")) != "boundary"
            or result.get("type") != "administrative"
            or result.get("address", {}).get("country_code") != "fi"
            or region_code != "FI-15"
            or str(tags.get("admin_level")) != "4"
            or "Pohjois-Savo" not in {result.get("name"), names.get("name"), names.get("name:fi")}):
        raise ValueError("Boundary is not the Finnish Pohjois-Savo region (administrative relation, FI-15)")
    polygon = shape(result["geojson"])
    if polygon.geom_type not in {"Polygon", "MultiPolygon"} or polygon.is_empty or not polygon.is_valid:
        raise ValueError("Pohjois-Savo boundary must be a valid, nonempty Polygon or MultiPolygon")
    # EPSG:4326 coordinates: X = longitude, Y = latitude. Sanity check only;
    # the actual downloaded polygon, including all rings, defines the region.
    west, south, east, north = polygon.bounds
    if not (19 < west < east < 32 and 59 < south < north < 71):
        raise ValueError("Boundary coordinates are outside Finland")
    return polygon


def fetch_boundary(metadata_dir: Path, nominatim_url: str, user_agent: str):
    cache_path = metadata_dir / "nominatim_boundary.json"
    if cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        # One serial query, no automatic retries. Keep the response for reuse.
        time.sleep(1.1)
        response = requests.get(
            nominatim_url.rstrip("/") + "/search",
            params={"state": "Pohjois-Savo", "country": "Finland", "countrycodes": "fi",
                    "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0,
                    "addressdetails": 1, "extratags": 1, "namedetails": 1, "limit": 5},
            headers={"User-Agent": user_agent, "Accept-Language": "fi"}, timeout=60,
        )
        response.raise_for_status()
        results = response.json()
        write_json(cache_path, results)
    matches = []
    for result in results:
        try:
            polygon = validate_boundary(result)
        except (ValueError, KeyError, TypeError):
            continue
        matches.append((result, polygon))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one verified Pohjois-Savo region; found {len(matches)}")
    result, polygon = matches[0]
    write_json(metadata_dir / "boundary.geojson", {
        "type": "FeatureCollection", "features": [{"type": "Feature",
        "properties": {"name": "Pohjois-Savo", "ISO3166-2": "FI-15", "admin_level": "4",
                       "osm_type": "relation", "osm_id": result["osm_id"],
                       "attribution": ATTRIBUTION}, "geometry": mapping(polygon)}]})
    return result, polygon


def validate_osm_graph(graph) -> None:
    if not graph.is_directed() or not graph.is_multigraph():
        raise ValueError("Expected a directed OSMnx MultiDiGraph")
    if graph.graph.get("simplified", False):
        raise ValueError("The OSM graph must not be simplified")
    if len(graph) == 0:
        raise ValueError("Downloaded OSM graph is empty")
    for node, data in graph.nodes(data=True):
        longitude, latitude = float(data["x"]), float(data["y"])
        if not (math.isfinite(longitude) and -180 <= longitude <= 180
                and math.isfinite(latitude) and -90 <= latitude <= 90):
            raise ValueError(f"Invalid coordinates at OSM node {node}")
    for u, v, key, data in graph.edges(keys=True, data=True):
        weight = float(data["length"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid length at OSM edge {(u, v, key)}")


def download(data_dir: Path, user_agent: str, nominatim_url: str, overpass_url: str) -> Path:
    raw_dir, metadata_dir = data_dir / "raw", data_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "pohjois_savo.graphml"
    metadata_path = metadata_dir / "download.json"
    if raw_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("status") != "success" or existing.get("graph_sha256") != file_sha256(raw_path):
            raise ValueError("Existing GraphML does not match successful download metadata")
        print(f"Using previously downloaded GraphML: {raw_path}")
        return raw_path
    metadata = {
        "region": "Pohjois-Savo, Finland", "iso3166_2": "FI-15",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress", "attribution": ATTRIBUTION,
        "license": "ODbL-1.0", "osmnx_version": ox.__version__,
        "graph_options": GRAPH_OPTIONS, "weight": "OSMnx length, metres",
        "user_agent": user_agent, "nominatim_url": nominatim_url,
        "overpass_url": overpass_url,
    }
    write_json(metadata_path, metadata)
    try:
        ox.settings.use_cache = True
        ox.settings.cache_folder = metadata_dir / "cache"
        ox.settings.http_user_agent = user_agent
        ox.settings.http_accept_language = "fi"
        ox.settings.overpass_rate_limit = True
        ox.settings.overpass_url = overpass_url
        ox.settings.requests_timeout = 180
        ox.settings.log_console = True
        ox.settings.all_oneway = False
        ox.settings.bidirectional_network_types = ["walk"]
        result, polygon = fetch_boundary(metadata_dir, nominatim_url, user_agent)
        metadata.update(boundary_osm_relation=int(result["osm_id"]),
                        boundary_display_name=result["display_name"],
                        boundary_geometry_type=polygon.geom_type)
        write_json(metadata_path, metadata)
        graph = ox.graph_from_polygon(polygon, **GRAPH_OPTIONS)
        validate_osm_graph(graph)
        temporary = raw_path.with_suffix(".graphml.tmp")
        ox.save_graphml(graph, filepath=temporary)
        digest = file_sha256(temporary)
        temporary.replace(raw_path)
        metadata.update(status="success", nodes=len(graph), directed_edges=graph.number_of_edges(),
                        graph_sha256=digest, crs=str(graph.graph["crs"]),
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                        failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
        raise
    print(f"Saved {len(graph)} nodes and {graph.number_of_edges()} directed edges to {raw_path}")
    print(ATTRIBUTION)
    return raw_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--user-agent", default=os.environ.get("OSM_USER_AGENT", USER_AGENT))
    parser.add_argument("--nominatim-url", default="https://nominatim.openstreetmap.org")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api")
    args = parser.parse_args()
    try:
        if not args.user_agent.strip():
            raise ValueError("A nonempty application User-Agent is required")
        download(args.data_dir, args.user_agent, args.nominatim_url, args.overpass_url)
    except (Exception, KeyboardInterrupt) as error:
        print(f"OSM download failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
