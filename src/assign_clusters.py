import geopandas as gpd
import pandas as pd
import numpy as np
import argparse
import os
import sys
from scipy.spatial import cKDTree

def main():
    parser = argparse.ArgumentParser(description="Assign buildings to nearest cluster models.")
    parser.add_argument("predictions_file", help="Path to predictions file (Shapefile or GeoJSON)")
    parser.add_argument("clusters_csv", help="Path to clusters CSV")
    parser.add_argument("output_csv", help="Path to output CSV for Blender import")
    parser.add_argument("--geospecific_geojson", help="Path to save geospecific building footprints", default=None)
    parser.add_argument("--max_dist", type=float, default=9999.0, help="Maximum fit distance to allow clustering. If exceeded, building is geospecific.")
    parser.add_argument("--max_levels_cluster", type=float, default=10.0, help="Maximum number of levels to allow clustering. If exceeded, building is geospecific.")
    
    args = parser.parse_args()
    
    # 1. Load Clusters
    print(f"Loading clusters from {args.clusters_csv}...")
    df_clusters = pd.read_csv(args.clusters_csv)
    
    # Check if levels are present
    has_levels = 'optimum_levels' in df_clusters.columns
    
    if has_levels:
        cluster_points = df_clusters[['optimum_width', 'optimum_height', 'optimum_levels']].values
    else:
        cluster_points = df_clusters[['optimum_width', 'optimum_height']].values
        
    cluster_dims_count = cluster_points.shape[1]
    cluster_tree = cKDTree(cluster_points)
    
    # Generate OBJ filenames for clusters
    cluster_obj_names = []
    for i, row in df_clusters.iterrows():
        w = row['optimum_width']
        h = row['optimum_height']
        
        if has_levels:
            l = int(row['optimum_levels'])
            fname = f"cluster_{i}_w{w:.2f}_h{h:.2f}_l{l}.obj"
        else:
            fname = f"cluster_{i}_w{w:.2f}_h{h:.2f}.obj"
            
        cluster_obj_names.append(fname)
        
    df_clusters['obj_filename'] = cluster_obj_names
    
    # 2. Load Predictions
    print(f"Loading predictions from {args.predictions_file}...")
    gdf_buildings = gpd.read_file(args.predictions_file)
    
    print(f"Loaded {len(gdf_buildings)} buildings.")
    
    if len(gdf_buildings) == 0:
        print("No buildings found in the input file.")
        return
        
    # Check for levels in input (Enhanced detection)
    levels_col = 'building:levels'
    possible_levels_cols = ['building:levels', 'building_levels', 'building_l', 'levels', 'L']
    detected_col = None
    for c in possible_levels_cols:
        if c in gdf_buildings.columns:
            detected_col = c
            break
            
    if detected_col and detected_col != levels_col:
        print(f"Found levels in column '{detected_col}'. Mapping to '{levels_col}'.")
        gdf_buildings[levels_col] = gdf_buildings[detected_col]
        
    if has_levels and levels_col not in gdf_buildings.columns:
        print(f"Warning: Clusters have levels but input file missing levels column. Assuming 1.")
        gdf_buildings[levels_col] = 1

    # 4. Assign Clusters
    print("Assigning nearest clusters...")
    
    # Extract width/height from predictions
    w_col = 'pred_width'
    h_col = 'pred_height'
    a_col = 'pred_angle'
    
    if h_col not in gdf_buildings.columns and 'pred_heigh' in gdf_buildings.columns:
        h_col = 'pred_heigh'
        
    required_cols = [w_col, h_col, a_col]
    for col in required_cols:
        if col not in gdf_buildings.columns:
            print(f"Error: Missing column '{col}' in file. Available: {gdf_buildings.columns.tolist()}")
            return
            
    # Prepare query points
    if has_levels:
        # Ensure numeric
        gdf_buildings[levels_col] = pd.to_numeric(gdf_buildings[levels_col], errors='coerce').fillna(1)
        query_points = gdf_buildings[[w_col, h_col, levels_col]].values
    else:
        query_points = gdf_buildings[[w_col, h_col]].values
    
    # Vectorized logic
    print("Calculating distances to all clusters...")
    cluster_dims = cluster_points # (M, D)
    building_dims = query_points  # (N, D)
    
    diff = building_dims[:, np.newaxis, :] - cluster_dims[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) # (N, M)
    
    # Check "Smaller or Equal" condition
    cond_w = cluster_dims[np.newaxis, :, 0] <= building_dims[:, np.newaxis, 0]
    cond_h = cluster_dims[np.newaxis, :, 1] <= building_dims[:, np.newaxis, 1]
    
    if has_levels:
        cond_l = cluster_dims[np.newaxis, :, 2] <= building_dims[:, np.newaxis, 2]
        is_smaller = cond_w & cond_h & cond_l
    else:
        is_smaller = cond_w & cond_h
    
    # Mask distances where cluster is NOT smaller
    masked_dist = np.where(is_smaller, dist_sq, np.inf)
    
    # Find best index (closest among smaller)
    best_idx = np.argmin(masked_dist, axis=1)
    min_vals = np.min(masked_dist, axis=1)
    
    # Handle Fallback
    fallback_mask = np.isinf(min_vals)
    fallback_count = np.sum(fallback_mask)
    
    # Identify Geospecific
    distances = np.sqrt(min_vals)
    
    if fallback_count > 0:
        fallback_indices = np.argmin(dist_sq[fallback_mask], axis=1)
        best_idx[fallback_mask] = fallback_indices
        distances[fallback_mask] = np.sqrt(np.min(dist_sq[fallback_mask], axis=1))
        
    indices = best_idx
    
    # Determine Clustered vs Geospecific
    is_geospecific = np.zeros(len(gdf_buildings), dtype=bool)
    
    if args.geospecific_geojson:
        # 1. Distance > max_dist
        mask_dist = distances > args.max_dist
        
        # 2. Levels > max_levels_cluster (New)
        mask_tall = np.zeros(len(gdf_buildings), dtype=bool)
        if has_levels:
             mask_tall = gdf_buildings[levels_col] > args.max_levels_cluster
        
        is_geospecific = mask_dist | mask_tall
        print(f"Identified {np.sum(mask_dist)} buildings with distance > {args.max_dist}.")
        print(f"Identified {np.sum(mask_tall)} buildings with levels > {args.max_levels_cluster}.")
        print(f"Total geospecific buildings: {np.sum(is_geospecific)}.")
        
    # 5. Prepare Output
    output_data = []
    
    # Indices of clustered
    clustered_indices = np.where(~is_geospecific)[0]
    
    for i in clustered_indices:
        idx = indices[i]
        dist = distances[i]
        
        building = gdf_buildings.iloc[i]
        cluster_file = df_clusters.iloc[idx]['obj_filename']
        cluster_levels = int(df_clusters.iloc[idx]['optimum_levels']) if has_levels else 1
        
        centroid = building.geometry.centroid
        x = centroid.x
        y = centroid.y
        z = 0
        
        try:
             angle_deg = building[a_col]
        except:
             angle_deg = 0
        
        entry = {
            'x': x,
            'y': y,
            'z': z,
            'angle_deg': angle_deg,
            'obj_filename': cluster_file,
            'pred_width': building[w_col],
            'pred_height': building[h_col],
            'fit_dist': dist
        }
        if has_levels:
            entry['pred_levels'] = building[levels_col]
            entry['cluster_levels'] = cluster_levels
            
        output_data.append(entry)
        
    df_out = pd.DataFrame(output_data)
    
    print(f"Saving {len(df_out)} clustered instances to {args.output_csv}...")
    df_out.to_csv(args.output_csv, index=False)
    
    # Save Geospecific
    if args.geospecific_geojson:
        geospecific_gdf = gdf_buildings[is_geospecific].copy()
        try:
            if os.path.exists(args.geospecific_geojson):
                try:
                    os.remove(args.geospecific_geojson)
                except PermissionError:
                    print(f"Warning: Could not remove {args.geospecific_geojson}. It might be locked.")

            if len(geospecific_gdf) > 0:
                out_ext = os.path.splitext(args.geospecific_geojson)[1].lower()
                driver = "GeoJSON"
                if out_ext == ".gpkg":
                    driver = "GPKG"
                elif out_ext == ".shp":
                    driver = "ESRI Shapefile"

                print(f"Saving {len(geospecific_gdf)} geospecific footprints to {args.geospecific_geojson} (Driver: {driver})...")
                try:
                    geospecific_gdf.to_file(args.geospecific_geojson, driver=driver)
                except Exception as e:
                    print(f"Error saving with default engine: {e}. Trying 'fiona'...")
                    geospecific_gdf.to_file(args.geospecific_geojson, driver=driver, engine='fiona')
            else:
                print("No geospecific buildings found.")
                with open(args.geospecific_geojson, 'w') as f:
                    f.write('{"type": "FeatureCollection", "features": []}')
        except PermissionError:
            print("\n" + "!"*60)
            print(f"CRITICAL ERROR: Permission denied when writing to {args.geospecific_geojson}")
            print("Please ensure the file is NOT open in QGIS, Blender, or any other program.")
            print("!"*60 + "\n")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error saving geospecific footprints: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
