import argparse
import os
import subprocess
import sys
import shutil
import time
import pyogrio
import pandas as pd
import geopandas as gpd

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
        # Use shell=True specifically on Windows if needing to resolve PATH or similar, 
        # but generally safer without if full path known.
        # Since we are calling python scripts, let's use sys.executable
        subprocess.check_call(cmd)
        print(f"-> {description} Completed Successfully.")
    except subprocess.CalledProcessError as e:
        print(f"-> Error: {description} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full MOBB pipeline: Predict -> Histograms -> Optimize -> Assign -> OBJ Models.")
    
    parser.add_argument("input_file", help="Path to input building footprints (Shapefile, GeoJSON, or GeoPackage)")
    parser.add_argument("model_file", help="Path to trained Random Forest model (.pkl)")
    parser.add_argument("output_dir", help="Directory to store all outputs")
    
    # Optional args passed to substeps
    # Default texture path relative to script directory
    default_texture = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures", "facade.png")
    parser.add_argument("--texture", default=default_texture, help="Path to facade texture image (default: textures/facade.png)")
    parser.add_argument("--roof_texture", default="textures/roof.png", help="Path to roof texture image (default: textures/roof.png)")
    parser.add_argument("--no-texture", action="store_true", help="Disable texture generation")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for optimization")
    parser.add_argument("--min_iou", type=float, default=0.0, help="Minimum IoU filter for clustering")
    parser.add_argument("--max_dist", type=float, default=5.0, help="Maximum fit distance (units) for clustering. Buildings exceeding this become geospecific.")
    parser.add_argument("--use_mobb", action="store_true", help="Calculate dimensions using MOBB instead of predicting")
    parser.add_argument("--merge", action="store_true", help="Initial step to merge contiguous polygons")
    parser.add_argument("--layer", help="Layer name to read from multi-layer files (e.g. GPKG)")
    
    args = parser.parse_args()
    
    start_time_total = time.time()
    kpis = {}
    
    # Handle --no-texture flag
    if args.no_texture:
        args.texture = None
        args.roof_texture = None
    
    # 0. Setup
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
    
    # Check if input is already predictions
    skip_predict = False
    try:
        # Read just a few rows to check columns
        gdf_check = gpd.read_file(args.input_file, rows=1)
        if 'pred_width' in gdf_check.columns:
            print("Input file appears to be existing predictions (found 'pred_width'). Skipping Step 1.")
            skip_predict = True
            # For GPKG we might need to be careful with copy if it's multi-layer, 
            # but usually the input_file to the pipeline is what we want.
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
            
        run_command(cmd_merge, "Step 0.5: Merge Contiguous Polygons")
        current_input = merged_input_file
        
        # Log merge reduction
        try:
            gdf_merged = gpd.read_file(merged_input_file)
            kpis['merged_count'] = len(gdf_merged)
            kpis['reduction_pct'] = (1 - (kpis['merged_count'] / kpis['input_count'])) * 100
        except:
            pass

    # 1. Predict
    # python src/predict_mobb.py input output --model model
    if not skip_predict:
        cmd_predict = [
            sys.executable,
            os.path.join(script_dir, "predict_mobb.py"),
            current_input,
            predictions_file,
            "--model", args.model_file
        ]
    # ...
        if args.use_mobb:
            cmd_predict.append("--use_mobb")
        if args.layer and not args.merge:
            # If we merged, the input to predict is the merged GeoJSON (single layer).
            # If we didn't merge, we need to pass the layer.
            cmd_predict.extend(["--layer", args.layer])
        
        # Purge output before Step 1
        purge_outputs([predictions_file])
        run_command(cmd_predict, "Step 1: Predict Dimensions")
    else:
        print("-> Step 1: Predict Dimensions SKIPPED (Input already processed).")
    
    # 2. Visualize Histograms
    # python src/vis_histograms.py input output
    # Purge output before Step 2
    purge_outputs([histograms_img])
    
    cmd_vis = [
        sys.executable,
        os.path.join(script_dir, "vis_histograms.py"),
        predictions_file,
        histograms_img
    ]
    run_command(cmd_vis, "Step 2: Visualize Histograms")
    
    # 3. Optimize Clusters
    # python src/optimize_clusters.py input output --n_clusters N --plot plot --min_iou IOU
    # Purge outputs before Step 3
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
    # python src/assign_clusters.py predictions clusters output --geospecific_geojson geojson --max_dist MAX
    # Purge outputs before Step 4
    purge_outputs([instances_csv, geospecific_geojson, geospecific_instances_csv])
    
    cmd_assign = [
        sys.executable,
        os.path.join(script_dir, "assign_clusters.py"),
        predictions_file,
        clusters_csv,
        instances_csv,
        "--geospecific_geojson", geospecific_geojson,
        "--max_dist", str(args.max_dist)
    ]
    run_command(cmd_assign, "Step 4: Assign Clusters")
    
    # 5. Create OBJ Models (Clustered)
    # python src/create_obj_models.py clusters output_dir --texture texture
    # Purge directory before Step 5
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
        
    run_command(cmd_create_obj, "Step 5: Create Clustered OBJ Models")
    
    # 6. Create Geospecific Models
    # python src/create_geospecific.py input output_dir output_csv
    # Only run if geospecific file exists and is not empty
    # Purge directory before Step 6
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
            
        run_command(cmd_geo, "Step 6: Create Geospecific Models")
    
    # 7. Reconstruct Geotypical Footprints
    # python src/reconstruct_geotypical_footprints.py instances_csv output_geojson
    geotypical_footprints_geojson = os.path.join(args.output_dir, "geotypical_footprints.geojson")
    
    # Purge output before Step 7
    purge_outputs([geotypical_footprints_geojson])
    
    cmd_footprints = [
        sys.executable,
        os.path.join(script_dir, "reconstruct_geotypical_footprints.py"),
        instances_csv,
        geotypical_footprints_geojson
    ]
    run_command(cmd_footprints, "Step 7: Reconstruct Geotypical Footprints")
    
    # --- Final KPI Collection ---
    try:
        # Prediction Quality (IoU)
        if os.path.exists(predictions_file):
            gdf_preds = gpd.read_file(predictions_file)
            if 'pred_iou' in gdf_preds.columns:
                kpis['avg_iou'] = gdf_preds['pred_iou'].mean()
        
        # Clustered Count
        if os.path.exists(instances_csv):
            df_inst = pd.read_csv(instances_csv)
            kpis['clustered_count'] = len(df_inst)
            
        # Geospecific Count
        if os.path.exists(geospecific_instances_csv):
            df_geo = pd.read_csv(geospecific_instances_csv)
            kpis['geospecific_count'] = len(df_geo)
            
        # Total time
        kpis['total_time_s'] = time.time() - start_time_total
    except Exception as e:
        print(f"Warning: Error collecting final KPIs: {e}")

    # --- Print Summary ---
    print("\n" + "="*50)
    print("           HYBRIDGT PIPELINE SUMMARY")
    print("="*50)
    
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-"*50)
    
    if 'input_count' in kpis:
        print(f"{'Input Buildings':<30} | {kpis['input_count']:<15}")
    
    if 'merged_count' in kpis:
        print(f"{'Buildings After Merge':<30} | {kpis['merged_count']:<15}")
        print(f"{'Merge Reduction':<30} | {kpis['reduction_pct']:.1f}%")
        
    if 'clustered_count' in kpis:
        print(f"{'Clustered Buildings':<30} | {kpis['clustered_count']:<15}")
    
    if 'geospecific_count' in kpis:
        print(f"{'Geospecific Buildings':<30} | {kpis['geospecific_count']:<15}")
        
    if 'avg_iou' in kpis:
        print(f"{'Average Fit IoU':<30} | {kpis['avg_iou']:.3f}")
        
    print("-"*50)
    print(f"{'Execution Mode':<30} | {'MOBB' if args.use_mobb else 'Prediction'}")
    print(f"{'Total Time':<30} | {kpis.get('total_time_s', 0):.2f}s")
    print("="*50)

    # --- Create Markdown Summary ---
    summary_md_path = os.path.join(args.output_dir, "summary.md")
    try:
        with open(summary_md_path, 'w') as f:
            f.write("# HybridGT Pipeline Execution Summary\n\n")
            f.write(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Execution Mode:** {'MOBB' if args.use_mobb else 'Prediction'}\n")
            f.write(f"- **Input File:** `{os.path.basename(args.input_file)}`\n")
            f.write(f"- **Total Time:** {kpis.get('total_time_s', 0):.2f}s\n\n")
            
            f.write("## Building Connectivity & Merging\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            if 'input_count' in kpis:
                f.write(f"| Initial Polygons | {kpis['input_count']} |\n")
            if 'merged_count' in kpis:
                f.write(f"| Polygons After Merge | {kpis['merged_count']} |\n")
                f.write(f"| Reduction | {kpis['reduction_pct']:.1f}% |\n")
            f.write("\n")
            
            f.write("## Clustering & Fitting Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            if 'clustered_count' in kpis:
                f.write(f"| Clustered (Geotypical) | {kpis['clustered_count']} |\n")
            if 'geospecific_count' in kpis:
                f.write(f"| Geospecific | {kpis['geospecific_count']} |\n")
            if 'avg_iou' in kpis:
                f.write(f"| Average Intersection over Union (IoU) | {kpis['avg_iou']:.3f} |\n")
            f.write("\n")
            
            f.write("## Output Files\n\n")
            f.write(f"- **Clustered Instances:** `instances.csv`\n")
            f.write(f"- **Geospecific Instances:** `geospecific_instances.csv`\n")
            f.write(f"- **Clustered Models:** `obj_models/`\n")
            f.write(f"- **Geospecific Models:** `geospecific_models/`\n")
            f.write(f"- **Geotypical Footprints:** `geotypical_footprints.geojson`\n")
            
        print(f"Detailed Markdown summary saved to: {summary_md_path}")
    except Exception as e:
        print(f"Warning: Could not save summary.md: {e}")

    print("\n=== Pipeline Completed Successfully ===")
    print(f"Intermediate files in: {args.output_dir}")
    print(f"Final Clustered Instances CSV: {instances_csv}")
    print(f"Final Geospecific Instances CSV: {geospecific_instances_csv}")
    print(f"Clustered OBJ Models: {obj_models_dir}")
    print(f"Geospecific OBJ Models: {geospecific_models_dir}")

if __name__ == "__main__":
    main()
