"""Feed genuine OSM extract records into OSMnx's polygon graph builder."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import osmium
import osmnx as ox
from osmnx import _overpass
from shapely import intersects_xy


def drive_filter():
    """Compile the pinned OSMnx drive filter; reject unsupported filter syntax."""
    expression = _overpass._get_network_filter("drive")
    token = re.compile(r'\["([^"\\]+)"(?:!~"([^"\\]*)")?\]')
    matches = list(token.finditer(expression))
    if "".join(match.group(0) for match in matches) != expression or not matches:
        raise ValueError("Unsupported OSMnx drive filter for local OSM extracts")
    clauses = [(match[1], re.compile(match[2]) if match[2] is not None else None)
               for match in matches]

    def accepts(tags):
        return all((key in tags if pattern is None
                    else key not in tags or pattern.search(tags[key]) is None)
                   for key, pattern in clauses)

    return expression, accepts


def extract_records(path: Path, polygon, work_dir: Path, metadata: dict):
    """Select complete drive ways touching the polygon at an original OSM node."""
    expression, accepts = drive_filter()
    ways = []
    required_nodes = set()
    west, south, east, north = polygon.bounds
    work_dir.mkdir(parents=True, exist_ok=True)

    # Validate the PBF header before creating the disk-backed location cache.
    with osmium.io.Reader(str(path), osmium.osm.NOTHING) as reader:
        metadata["osm_snapshot_utc"] = reader.header().get("osmosis_replication_timestamp")

    # Omitting a filename lets libosmium manage its own temporary disk index.
    # A named sparse_file_array retains an open file descriptor even after the
    # processor is deleted, preventing TemporaryDirectory cleanup on Windows.
    processor = (osmium.FileProcessor(path)
                 .with_locations("sparse_file_array")
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                 .with_filter(osmium.filter.KeyFilter("highway")))
    print("Reading original road ways from the OSM extract...", flush=True)
    for way in processor:
        tags = dict(way.tags)
        if not accepts(tags):
            continue
        refs, longitude, latitude = [], [], []
        for node in way.nodes:
            if not node.location.valid():
                raise ValueError(f"OSM extract lacks location for node {node.ref} of way {way.id}")
            refs.append(node.ref)
            longitude.append(node.lon)
            latitude.append(node.lat)
        candidates = [(x, y) for x, y in zip(longitude, latitude)
                      if west <= x <= east and south <= y <= north]
        if not candidates:
            continue
        xs, ys = zip(*candidates)
        if not intersects_xy(polygon, xs, ys).any():
            continue
        ways.append({"type": "way", "id": way.id, "nodes": refs, "tags": tags})
        required_nodes.update(refs)
    del processor

    print(f"Reading {len(required_nodes)} original nodes for {len(ways)} selected ways...", flush=True)
    nodes = []
    found_nodes = set()
    processor = (osmium.FileProcessor(path, osmium.osm.NODE)
                 .with_filter(osmium.filter.IdFilter(required_nodes)))
    for node in processor:
        if node.id in found_nodes:
            raise ValueError("OSM history extracts are unsupported: duplicate node versions")
        found_nodes.add(node.id)
        nodes.append({"type": "node", "id": node.id, "lon": node.lon, "lat": node.lat,
                      "tags": dict(node.tags)})
    if found_nodes != required_nodes:
        raise ValueError("OSM extract is missing nodes referenced by selected roads")
    if not ways:
        raise ValueError("OSM extract contains no drive ways within the region")
    metadata.update(drive_filter=expression, selected_osm_ways=len(ways), selected_osm_nodes=len(nodes))
    return {"elements": nodes + ways}


def graph_from_extract(path: Path, polygon, work_dir: Path, graph_options: dict):
    """Use the normal OSMnx topology, length, direction and polygon operations.

    The pinned OSMnx version has no public PBF input to graph_from_polygon.
    Replace only its record supplier for this sequential call, restoring it even
    on failure. These records come from the file, never from an Overpass cache.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {"type": "osm_extract", "file": path.name, "sha256": digest.hexdigest()}

    def local_records(buffered_polygon, network_type, custom_filter):
        if network_type != "drive" or custom_filter is not None:
            raise ValueError("Local OSM input supports only the standard drive filter")
        yield extract_records(path, buffered_polygon, work_dir, metadata)

    original = _overpass._download_overpass_network
    try:
        _overpass._download_overpass_network = local_records
        graph = ox.graph_from_polygon(polygon, **graph_options)
    finally:
        _overpass._download_overpass_network = original
    return graph, metadata
