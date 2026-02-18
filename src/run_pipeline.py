import argparse
import os
import subprocess
import sys
import shutil
import time
import pyogrio
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union

def purge_outputs(filepaths):
    """
    Attempts to delete the specified files to prevent stale data usage.
    Exits if a file is locked and cannot be deleted.
    """
    for fp in filepaths:
        if fp and os.path.exists(fp):
            try:
                if os.path.isdir(fp):
                    shutil.rmtree(fp)
                else:
                    os.remove(fp)
            except PermissionError:
                print("\n" + "!"*60)
                print(f"CRITICAL ERROR: Could not purge stale output: {fp}")
                print("The file is locked by another process (e.g., QGIS, Blender).")
                print("Please close any programs using this file and try again.")
                print("!"*60 + "\n")
                sys.exit(1)
            except Exception as e:
                print(f"Warning: Could not delete {fp}: {e}")

def run_command(cmd, description):
    """Runs a command and prints status."""
    print(f"\n--- {description} ---")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"-> {description} Completed Successfully.")
    except subprocess.CalledProcessError as e:
        print(f"-> Error: {description} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def download_overpass(oaci_code, output_file):
    """Download building footprints within 25 km of an airport via its ICAO code using the Overpass API."""
    query = f"""[out:json][timeout:900];

// 1. Find the airport feature using the ICAO code
nwr["icao"="{oaci_code}"]->.airport;

// 2. Search for buildings within 25km of that specific result
(
  nwr["building"](around.airport:25000);
);

// 3. Output full footprints
out geom;
"""
    print(f"Querying Overpass API for ICAO code: {oaci_code} ...")
    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(url, data={"data": query}, timeout=960)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Overpass API request failed: {e}")
        sys.exit(1)

    data = response.json()
    elements = data.get("elements", [])
    print(f"Received {len(elements)} elements from Overpass API.")

    records = []
    for el in elements:
        etype = el.get("type")
        tags = el.get("tags", {})

        if etype == "way":
            geom_nodes = el.get("geometry", [])
            if len(geom_nodes) >= 3:
                coords = [(n["lon"], n["lat"]) for n in geom_nodes]
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and not poly.is_empty:
                        rec = {"geometry": poly}
                        rec.update(tags)
                        records.append(rec)
                except Exception:
                    pass

        elif etype == "relation":
            outer_rings = []
            for member in el.get("members", []):
                if member.get("role") == "outer" and "geometry" in member:
                    ring = [(n["lon"], n["lat"]) for n in member["geometry"]]
                    if len(ring) >= 3:
                        outer_rings.append(ring)
            if outer_rings:
                try:
                    parts = [Polygon(c) for c in outer_rings if len(c) >= 3]
                    poly = unary_union(parts)
                    if poly.is_valid and not poly.is_empty:
                        rec = {"geometry": poly}
                        rec.update(tags)
                        records.append(rec)
                except Exception:
                    pass

    if not records:
        print(f"Error: No building footprints found for ICAO code '{oaci_code}'. "
              "Check that the code is correct and the airport has an 'icao' tag in OpenStreetMap.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"Saved {len(gdf)} building footprints to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run the full MOBB pipeline: Predict -> Histograms -> Optimize -> Assign -> OBJ Models.")
    
    parser.add_argument("input_file", nargs="?", default=None,
                        help="Path to input building footprints (Shapefile, GeoJSON, or GeoPackage). "
                             "Not required when --oaci is used.")
    parser.add_argument("model_file", help="Path to trained Random Forest model (.pkl)")
    parser.add_argument("output_dir", help="Directory to store all outputs")
    
    # Optional args passed to substeps
    default_texture = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures", "facade.png")
    parser.add_argument("--texture", default=default_texture, help="Path to facade texture image (default: textures/facade.png)")
    parser.add_argument("--roof_texture", default="textures/roof.png", help="Path to roof texture image (default: textures/roof.png)")
    parser.add_argument("--emissive_texture", default="textures/emmissive.png", help="Path to emissive texture image (default: textures/emmissive.png)")
    parser.add_argument("--no-texture", action="store_true", help="Disable texture generation")
    
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for optimization")
    parser.add_argument("--min_iou", type=float, default=0.0, help="Minimum IoU filter for clustering")
    parser.add_argument("--max_dist", type=float, default=5.0, help="Maximum fit distance (units) for clustering. Buildings exceeding this become geospecific.")
    parser.add_argument("--min_area", type=float, default=10.0, help="Minimum footprint area to keep (default: 10m2)")
    parser.add_argument("--max_levels_cluster", type=float, default=10.0, help="Maximum levels allowed for clustering. Above this, buildings are geospecific.")
    
    parser.add_argument("--use_mobb", action="store_true", help="Calculate dimensions using MOBB instead of predicting")
    parser.add_argument("--merge", action="store_true", help="Initial step to merge contiguous polygons")
    parser.add_argument("--layer", help="Layer name to read from multi-layer files (e.g. GPKG)")
    parser.add_argument("--levels_col", help="Name of the attribute column containing building level data (e.g. 'building:levels', 'num_floors')")
    parser.add_argument("--oaci", metavar="ICAO_CODE",
                        help="ICAO airport code (e.g. WMSA). Downloads building footprints within 25 km "
                             "of the airport from the Overpass API and uses them as pipeline input. "
                             "Mutually exclusive with providing input_file directly.")

    args = parser.parse_args()

    # Validate: exactly one of input_file or --oaci must be provided
    if args.oaci and args.input_file:
        parser.error("Provide either input_file or --oaci, not both.")
    if not args.oaci and not args.input_file:
        parser.error("One of input_file or --oaci is required.")
    
    start_time_total = time.time()
    kpis = {}
    
    # Handle --no-texture flag
    if args.no_texture:
        args.texture = None
        args.roof_texture = None
        args.emissive_texture = None
    
    # 0. Setup
    # If --oaci is given, download building footprints from Overpass API first
    if args.oaci:
        os.makedirs(args.output_dir, exist_ok=True)
        overpass_file = os.path.join(args.output_dir, f"overpass_{args.oaci}.geojson")
        download_overpass(args.oaci, overpass_file)
        args.input_file = overpass_file

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading input footprints: {args.input_file}")
    
    # Check for multi-layer GPKG
    layer = args.layer
    if args.input_file.lower().endswith(".gpkg"):
        try:
            layers = pyogrio.list_layers(args.input_file)
            layer_names = layers[:, 0] if len(layers) > 0 else []
            
            if len(layer_names) > 1 and not layer:
                print("\n" + "!"*60)
                print(f"WARNING: Multiple layers detected in {os.path.basename(args.input_file)}:")
                for ln in layer_names:
                    print(f"  - {ln}")
                print(f"No --layer specified. Using the first layer: '{layer_names[0]}'")
                print("!"*60 + "\n")
                layer = layer_names[0]
            elif layer and layer not in layer_names:
                print(f"Error: Layer '{layer}' not found in {args.input_file}. Available: {list(layer_names)}")
                sys.exit(1)
        except Exception as e:
            print(f"Warning: Could not list layers in {args.input_file}: {e}")

    try:
        gdf_input = gpd.read_file(args.input_file, layer=layer)
        kpis['input_count'] = len(gdf_input)
    except Exception as e:
        print(f"Error reading input: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Define paths
    predictions_file = os.path.join(args.output_dir, "predictions.geojson")
    histograms_img = os.path.join(args.output_dir, "histograms.png")
    clusters_csv = os.path.join(args.output_dir, "clusters.csv")
    clusters_plot = os.path.join(args.output_dir, "clusters_plot.png")
    instances_csv = os.path.join(args.output_dir, "instances.csv")
    obj_models_dir = os.path.join(args.output_dir, "obj_models")
    merged_input_file = os.path.join(args.output_dir, "merged_footprints.geojson")
    
    # Geospecific paths
    geospecific_geojson = os.path.join(args.output_dir, "geospecific_buildings.geojson")
    geospecific_models_dir = os.path.join(args.output_dir, "geospecific_models")
    geospecific_instances_csv = os.path.join(args.output_dir, "geospecific_instances.csv")
    
    # Scripts location
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Auto-train model if the model file is not present
    if not os.path.exists(args.model_file):
        print(f"\nModel not found at '{args.model_file}'. Auto-training on input data ...")
        sys.path.insert(0, script_dir)
        from train_mobb import train_model
        result = train_model(gdf_input, args.model_file)
        if result is None:
            print("Error: Auto-training failed. Please supply a pre-trained model or provide "
                  "more building data (at least 10 valid polygons required).")
            sys.exit(1)
        print(f"-> Auto-training complete. Model saved to '{args.model_file}'.\n")

    # Check if input is already predictions
    skip_predict = False
    try:
        gdf_check = gpd.read_file(args.input_file, rows=1)
        if 'pred_width' in gdf_check.columns:
            print("Input file appears to be existing predictions (found 'pred_width'). Skipping Step 1.")
            skip_predict = True
            shutil.copy2(args.input_file, predictions_file)
    except Exception as e:
        print(f"Warning: Could not check input columns: {e}. assuming raw footprints.")

    # 0.5 Merge Footprints (Optional)
    current_input = args.input_file
    if args.merge:
        cmd_merge = [
            sys.executable,
            os.path.join(script_dir, "merge_footprints.py"),
            args.input_file,
            merged_input_file
        ]
        if args.layer:
            cmd_merge.extend(["--layer", args.layer])
        if args.levels_col:
            cmd_merge.extend(["--levels_col", args.levels_col])

        run_command(cmd_merge, "Step 0.5: Merge Contiguous Polygons")
        current_input = merged_input_file
        
        try:
            gdf_merged = gpd.read_file(merged_input_file)
            kpis['merged_count'] = len(gdf_merged)
            kpis['reduction_pct'] = (1 - (kpis['merged_count'] / kpis['input_count'])) * 100
        except:
            pass

    # 1. Predict
    if not skip_predict:
        cmd_predict = [
            sys.executable,
            os.path.join(script_dir, "predict_mobb.py"),
            current_input,
            predictions_file,
            "--model", args.model_file,
            "--min_area", str(args.min_area)
        ]
        if args.use_mobb:
            cmd_predict.append("--use_mobb")
        if args.layer and not args.merge:
            cmd_predict.extend(["--layer", args.layer])
        if args.levels_col:
            cmd_predict.extend(["--levels_col", args.levels_col])
        
        purge_outputs([predictions_file])
        run_command(cmd_predict, "Step 1: Predict Dimensions")
    else:
        print("-> Step 1: Predict Dimensions SKIPPED (Input already processed).")
    
    # 2. Visualize Histograms
    purge_outputs([histograms_img])
    cmd_vis = [
        sys.executable,
        os.path.join(script_dir, "vis_histograms.py"),
        predictions_file,
        histograms_img
    ]
    run_command(cmd_vis, "Step 2: Visualize Histograms")
    
    # 3. Optimize Clusters
    purge_outputs([clusters_csv, clusters_plot])
    cmd_optimize = [
        sys.executable,
        os.path.join(script_dir, "optimize_clusters.py"),
        predictions_file,
        clusters_csv,
        "--n_clusters", str(args.n_clusters),
        "--plot", clusters_plot,
        "--min_iou", str(args.min_iou)
    ]
    run_command(cmd_optimize, "Step 3: Optimize Clusters")
    
    # 4. Assign Clusters
    purge_outputs([instances_csv, geospecific_geojson, geospecific_instances_csv])
    cmd_assign = [
        sys.executable,
        os.path.join(script_dir, "assign_clusters.py"),
        predictions_file,
        clusters_csv,
        instances_csv,
        "--geospecific_geojson", geospecific_geojson,
        "--max_dist", str(args.max_dist),
        "--max_levels_cluster", str(args.max_levels_cluster)
    ]
    run_command(cmd_assign, "Step 4: Assign Clusters")
    
    # 5. Create OBJ Models (Clustered)
    purge_outputs([obj_models_dir])
    cmd_create_obj = [
        sys.executable,
        os.path.join(script_dir, "create_obj_models.py"),
        clusters_csv,
        obj_models_dir
    ]
    if args.texture:
        cmd_create_obj.extend(["--texture", args.texture])
    if args.roof_texture:
        cmd_create_obj.extend(["--roof_texture", args.roof_texture])
    if args.emissive_texture:
        cmd_create_obj.extend(["--emissive_texture", args.emissive_texture])
        
    run_command(cmd_create_obj, "Step 5: Create Clustered OBJ Models")
    
    # 6. Create Geospecific Models
    purge_outputs([geospecific_models_dir])
    if os.path.exists(geospecific_geojson):
        cmd_geo = [
            sys.executable,
            os.path.join(script_dir, "create_geospecific.py"),
            geospecific_geojson,
            geospecific_models_dir,
            geospecific_instances_csv
        ]
        if args.texture:
            cmd_geo.extend(["--texture", args.texture])
        if args.roof_texture:
            cmd_geo.extend(["--roof_texture", args.roof_texture])
        if args.emissive_texture:
            cmd_geo.extend(["--emissive_texture", args.emissive_texture])
            
        run_command(cmd_geo, "Step 6: Create Geospecific Models")
    
    # 7. Reconstruct Geotypical Footprints
    geotypical_footprints_geojson = os.path.join(args.output_dir, "geotypical_footprints.geojson")
    purge_outputs([geotypical_footprints_geojson])
    cmd_footprints = [
        sys.executable,
        os.path.join(script_dir, "reconstruct_geotypical_footprints.py"),
        instances_csv,
        geotypical_footprints_geojson
    ]
    run_command(cmd_footprints, "Step 7: Reconstruct Geotypical Footprints")
    
    # KPIs and Summary
    try:
        if os.path.exists(predictions_file):
            gdf_preds = gpd.read_file(predictions_file)
            if 'pred_iou' in gdf_preds.columns:
                kpis['avg_iou'] = gdf_preds['pred_iou'].mean()
        if os.path.exists(instances_csv):
            df_inst = pd.read_csv(instances_csv)
            kpis['clustered_count'] = len(df_inst)
        if os.path.exists(geospecific_instances_csv):
            df_geo = pd.read_csv(geospecific_instances_csv)
            kpis['geospecific_count'] = len(df_geo)
        kpis['total_time_s'] = time.time() - start_time_total
    except Exception as e:
        print(f"Warning: Error collecting final KPIs: {e}")

    # Print Summary ...
    print("\n" + "="*50)
    print("           HYBRIDGT PIPELINE SUMMARY")
    print("="*50)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-"*50)
    if 'input_count' in kpis:
        print(f"{'Input Buildings':<30} | {kpis['input_count']:<15}")
    if 'clustered_count' in kpis:
        print(f"{'Clustered Buildings':<30} | {kpis['clustered_count']:<15}")
    if 'geospecific_count' in kpis:
        print(f"{'Geospecific Buildings':<30} | {kpis['geospecific_count']:<15}")
    print(f"{'Total Time':<30} | {kpis.get('total_time_s', 0):.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()
