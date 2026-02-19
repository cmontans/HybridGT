"""
HybridGT Pipeline Server
FastAPI backend: node registry, pipeline CRUD, WebSocket execution engine.
"""
import asyncio
import json
import os
import sys
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
NODES_DIR = SRC_DIR / "nodes"
PIPELINES_DIR = BASE_DIR / "pipelines"
CACHE_DIR = BASE_DIR / ".hybridgt_cache"

PIPELINES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="HybridGT Pipeline Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

def load_node_registry() -> Dict[str, Any]:
    """Scan src/nodes/*.node.yaml and return dict keyed by node id."""
    registry: Dict[str, Any] = {}
    for yaml_path in sorted(NODES_DIR.glob("*.node.yaml")):
        with open(yaml_path) as f:
            defn = yaml.safe_load(f)
        registry[defn["id"]] = defn
    return registry


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/nodes")
def get_nodes():
    """Return all node definitions grouped as a list."""
    registry = load_node_registry()
    return {"nodes": list(registry.values())}


@app.get("/api/pipelines")
def list_pipelines():
    names = [p.stem for p in sorted(PIPELINES_DIR.glob("*.json"))]
    return {"pipelines": names}


@app.get("/api/pipelines/{name}")
def get_pipeline(name: str):
    path = PIPELINES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    return json.loads(path.read_text())


@app.post("/api/pipelines/{name}")
async def save_pipeline(name: str, request_body: dict):
    path = PIPELINES_DIR / f"{name}.json"
    path.write_text(json.dumps(request_body, indent=2))
    return {"saved": name}


@app.delete("/api/pipelines/{name}")
def delete_pipeline(name: str):
    path = PIPELINES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    path.unlink()
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

PORT_TYPE_EXT: Dict[str, str] = {
    "geojson": "geojson",
    "csv":     "csv",
    "pkl":     "pkl",
    "png":     "png",
    "shp":     "shp",
    "gpkg":    "gpkg",
}

SOURCE_NODE_TYPES = {"file_input", "dir_input"}


def port_produces_ext(port_def: dict) -> str:
    produces = port_def.get("produces", "")
    return PORT_TYPE_EXT.get(produces, produces) or "out"


def output_path_for_port(run_id: str, node_id: str, port_id: str, ext: str) -> str:
    node_cache = CACHE_DIR / run_id / node_id
    node_cache.mkdir(parents=True, exist_ok=True)
    if ext:
        return str(node_cache / f"{port_id}.{ext}")
    # Directory port: return the directory path itself
    dir_path = node_cache / port_id
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)


def topological_sort(nodes: List[dict], edges: List[dict]) -> List[str]:
    """Kahn's algorithm. Returns ordered list of node ids. Raises on cycle."""
    adj: Dict[str, set] = defaultdict(set)
    in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if tgt not in adj[src]:
            adj[src].add(tgt)
            in_degree[tgt] += 1

    queue = deque(sorted([nid for nid, deg in in_degree.items() if deg == 0]))
    order: List[str] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for downstream in sorted(adj[nid]):
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    if len(order) != len(nodes):
        raise ValueError("Pipeline graph has a cycle — cannot execute")

    return order


def resolve_inputs(
    node_id: str,
    node_def: dict,
    edges: List[dict],
    output_paths: Dict[str, Dict[str, str]],
    node_params: Dict[str, Any],
) -> Dict[str, str]:
    """
    For each input port, resolve its value from:
    1. An upstream edge (takes precedence)
    2. A user-typed param value
    """
    resolved: Dict[str, str] = {}
    input_ports = node_def.get("ports", {}).get("inputs", [])

    for port in input_ports:
        port_id = port["id"]
        # Edge takes precedence
        for edge in edges:
            if edge["target"] == node_id and edge["targetHandle"] == port_id:
                src_node = edge["source"]
                src_port = edge["sourceHandle"]
                val = output_paths.get(src_node, {}).get(src_port, "")
                if val:
                    resolved[port_id] = val
                break
        else:
            # Fall back to param value (same id as port)
            if port_id in node_params and node_params[port_id] not in (None, ""):
                resolved[port_id] = str(node_params[port_id])

    return resolved


def build_command(
    node_def: dict,
    input_values: Dict[str, str],
    output_values: Dict[str, str],
    node_params: Dict[str, Any],
) -> List[str]:
    """Construct the subprocess command list from the node definition."""
    cmd_spec = node_def["command"]
    script_path = str(SRC_DIR / cmd_spec["script"])
    cmd = [sys.executable, script_path]

    # Positional arguments (in declared order)
    for pos in cmd_spec.get("positional", []):
        port_id = pos["port"]
        if port_id in input_values:
            cmd.append(input_values[port_id])
        elif port_id in output_values:
            cmd.append(output_values[port_id])
        else:
            raise ValueError(
                f"Cannot resolve positional port '{port_id}' for node '{node_def['id']}'"
            )

    # Flag arguments
    for flag_spec in cmd_spec.get("flags", []):
        flag = flag_spec["flag"]
        flag_type = flag_spec.get("type", "value")  # "value" | "flag"
        omit_if_null = flag_spec.get("omit_if_null", True)

        # Determine value source: param or output port
        if "param" in flag_spec:
            value = node_params.get(flag_spec["param"])
        elif "port" in flag_spec:
            # This flag references an output port (e.g. --geospecific_geojson)
            value = output_values.get(flag_spec["port"])
        else:
            continue

        # Boolean / store_true flags
        if flag_type == "flag":
            if value:
                cmd.append(flag)
            continue

        # Null / empty value handling
        if value is None or value == "":
            if omit_if_null:
                continue
            else:
                continue  # always skip truly empty values

        cmd.extend([flag, str(value)])

    return cmd


# ---------------------------------------------------------------------------
# WebSocket execution engine
# ---------------------------------------------------------------------------

@app.websocket("/ws/run")
async def run_pipeline(ws: WebSocket):
    await ws.accept()
    run_id = str(uuid.uuid4())[:8]

    # Receive graph JSON
    try:
        raw = await ws.receive_text()
        graph = json.loads(raw)
    except Exception as e:
        await ws.send_json({"type": "pipeline_error", "message": f"Invalid graph JSON: {e}"})
        return

    nodes_list: List[dict] = graph.get("nodes", [])
    edges_list: List[dict] = graph.get("edges", [])

    if not nodes_list:
        await ws.send_json({"type": "pipeline_error", "message": "Graph has no nodes"})
        return

    registry = load_node_registry()

    # Pre-build output_paths for every node
    output_paths: Dict[str, Dict[str, str]] = {}

    for node in nodes_list:
        ntype = node["type"]
        nid = node["id"]
        params = node.get("params", {})

        if ntype in SOURCE_NODE_TYPES:
            # Source nodes expose their typed path directly
            output_paths[nid] = {"value": params.get("path", "")}
            continue

        node_def = registry.get(ntype)
        if not node_def:
            await ws.send_json({
                "type": "pipeline_error",
                "message": f"Unknown node type '{ntype}'"
            })
            return

        output_paths[nid] = {}
        for port in node_def.get("ports", {}).get("outputs", []):
            ptype = port.get("type", "file")
            if ptype == "directory":
                path = output_path_for_port(run_id, nid, port["id"], "")
            else:
                ext = port_produces_ext(port)
                path = output_path_for_port(run_id, nid, port["id"], ext)
            output_paths[nid][port["id"]] = path

    # Topological sort
    try:
        order = topological_sort(nodes_list, edges_list)
    except ValueError as e:
        await ws.send_json({"type": "pipeline_error", "message": str(e)})
        return

    await ws.send_json({"type": "pipeline_start", "run_id": run_id, "order": order})

    # Execute each node in order
    for node_id in order:
        node = next(n for n in nodes_list if n["id"] == node_id)
        ntype = node["type"]

        if ntype in SOURCE_NODE_TYPES:
            await ws.send_json({"type": "node_skip", "node_id": node_id, "reason": "source"})
            continue

        node_def = registry[ntype]
        node_params = node.get("params", {})

        # Resolve inputs
        try:
            input_values = resolve_inputs(
                node_id, node_def, edges_list, output_paths, node_params
            )
        except Exception as e:
            await ws.send_json({
                "type": "node_error",
                "node_id": node_id,
                "message": f"Input resolution failed: {e}",
            })
            return

        # Build command
        try:
            cmd = build_command(
                node_def,
                input_values,
                output_paths[node_id],
                node_params,
            )
        except Exception as e:
            await ws.send_json({
                "type": "node_error",
                "node_id": node_id,
                "message": f"Command build failed: {e}",
            })
            return

        await ws.send_json({
            "type": "node_start",
            "node_id": node_id,
            "command": " ".join(cmd),
        })

        # Run subprocess with streaming output
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )

            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    await ws.send_json({"type": "log", "node_id": node_id, "text": line})

            await proc.wait()

        except Exception as e:
            await ws.send_json({
                "type": "node_error",
                "node_id": node_id,
                "message": f"Subprocess exception: {e}",
            })
            return

        if proc.returncode != 0:
            await ws.send_json({
                "type": "node_error",
                "node_id": node_id,
                "exit_code": proc.returncode,
                "message": f"Process exited with code {proc.returncode}",
            })
            return

        await ws.send_json({
            "type": "node_complete",
            "node_id": node_id,
            "outputs": output_paths[node_id],
        })

    await ws.send_json({"type": "pipeline_complete", "run_id": run_id})


# ---------------------------------------------------------------------------
# Serve built React frontend (production)
# ---------------------------------------------------------------------------
_ui_dist = BASE_DIR / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
