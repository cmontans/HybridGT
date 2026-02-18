import geopandas as gpd
import pandas as pd
import argparse
import os
import sys
import pyogrio
from shapely.geometry import Polygon, MultiPolygon

def remove_small_holes(geom, threshold_ratio=0.1):
    """
    Removes holes (interior rings) from a polygon if their area is less than 
    a threshold percentage of the exterior area.
    
    Args:
        geom: A shapely Polygon or MultiPolygon.
        threshold_ratio: Max area of hole relative to exterior area to be removed.
    """
    if geom is None or geom.is_empty:
        return geom
        
    if isinstance(geom, MultiPolygon):
        # Process each polygon in the multi-polygon
        return MultiPolygon([remove_small_holes(p, threshold_ratio) for p in geom.geoms])
    
    if geom.geom_type != 'Polygon':
        return geom

    # Polygon case
    exterior = geom.exterior
    interiors = []
    
    # Area of the exterior boundary (total area if no holes)
    total_area = Polygon(exterior).area
    
    for hole in geom.interiors:
        hole_poly = Polygon(hole)
        if hole_poly.area / total_area >= threshold_ratio:
            interiors.append(hole)
            
    return Polygon(exterior, interiors)

def merge_contiguous_polygons(input_path, output_path, buffer_dist=0.01, layer=None, levels_col=None):
    """
    Merges contiguous or overlapping polygons in a GeoJSON/Shapefile/GeoPackage.
    
    Args:
        input_path: Path to input building footprints.
        output_path: Path to save merged footprints.
        buffer_dist: Small buffer distance to ensure touching polygons intersect.
        layer: Specific layer to read from a multi-layer file (e.g., GPKG).
    """
    # Check for multi-layer GPKG
    if input_path.lower().endswith(".gpkg"):
        try:
            layers = pyogrio.list_layers(input_path)
            # layers is a numpy array of [name, type]
            layer_names = layers[:, 0] if len(layers) > 0 else []
            
            if len(layer_names) > 1 and not layer:
                print("\n" + "!"*60)
                print(f"WARNING: Multiple layers detected in {os.path.basename(input_path)}:")
                for ln in layer_names:
                    print(f"  - {ln}")
                print(f"No --layer specified. Using the first layer: '{layer_names[0]}'")
                print("!"*60 + "\n")
                layer = layer_names[0]
            elif layer and layer not in layer_names:
                print(f"Error: Layer '{layer}' not found in {input_path}. Available: {list(layer_names)}")
                sys.exit(1)
        except Exception as e:
            print(f"Warning: Could not list layers in {input_path}: {e}")

    print(f"Loading footprints from {input_path} (Layer: {layer if layer else 'default'})...")
    gdf = gpd.read_file(input_path, layer=layer)
    
    if len(gdf) == 0:
        print("Empty input file.")
        gdf.to_file(output_path, driver='GeoJSON')
        return

    print(f"Initial feature count: {len(gdf)}")

    # Ensure we are in a metric CRS for buffering if it's geographic
    original_crs = gdf.crs
    if original_crs and original_crs.is_geographic:
        # Reproject to UTM Zone 30N (default for this project) or similar for buffering
        gdf = gdf.to_crs("EPSG:32630")
    
    # 1. Apply a very small buffer to ensure touching boundaries overlap
    print(f"Applying small buffer ({buffer_dist}m) for robust merging...")
    buffered = gdf.geometry.buffer(buffer_dist)
    
    # 2. Perform Unary Union
    print("Performing union of overlapping geometries...")
    union_poly = buffered.unary_union
    
    # 3. Explode MultiPolygon back into individual features
    print("Exploding merged results...")
    if union_poly.geom_type == 'Polygon':
        merged_gdf = gpd.GeoDataFrame(geometry=[union_poly], crs=gdf.crs)
    else:
        # MultiPolygon or GeometryCollection
        merged_geoms = []
        if hasattr(union_poly, 'geoms'):
            for g in union_poly.geoms:
                if g.geom_type == 'Polygon':
                    merged_geoms.append(g)
                elif g.geom_type == 'MultiPolygon':
                    merged_geoms.extend(list(g.geoms))
        else:
            merged_geoms = [union_poly]
            
        merged_gdf = gpd.GeoDataFrame(geometry=merged_geoms, crs=gdf.crs)

    # 3.5 Remove Small Holes
    print("Removing holes smaller than 10% of polygon area...")
    merged_gdf.geometry = merged_gdf.geometry.apply(lambda g: remove_small_holes(g, 0.1))

    # 4. Remove the buffer (negative buffer) to restore original size
    # Actually, if we buffer by 0.01 and then -0.01, we might lose tiny details.
    # But for building footprints, 1cm is usually negligible.
    print(f"Removing buffer (-{buffer_dist}m)...")
    merged_gdf.geometry = merged_gdf.geometry.buffer(-buffer_dist)
    
    # --- Attribute Preservation (Levels/Height) ---
    print("Preserving building attributes (levels/height) using spatial join...")
    
    # 1. Detect level-related columns in original data
    possible_cols = ['building:levels', 'building_levels', 'building_l', 'levels', 'L', 'height', 'ele', 'altitude']
    if levels_col and levels_col not in possible_cols:
        possible_cols.insert(0, levels_col)
    detected_cols = [c for c in possible_cols if c in gdf.columns]
    
    if detected_cols:
        # Give merged_gdf a unique index for grouping
        merged_gdf['temp_idx'] = merged_gdf.index
        
        # Spatial join: Which original buildings fall into which merged footprint?
        # Use simple geometry subset for join to avoid bloat
        joined = gpd.sjoin(
            merged_gdf[['geometry', 'temp_idx']], 
            gdf[detected_cols + ['geometry']], 
            how='left', 
            predicate='intersects'
        )
        
        # 2. Aggregate attributes by merged footprint (take max)
        print(f"Aggregating {len(detected_cols)} columns from constituent buildings...")
        agg_map = {}
        for col in detected_cols:
            # Ensure numeric for max()
            joined[col] = pd.to_numeric(joined[col], errors='coerce')
            agg_map[col] = 'max'
            
        attr_data = joined.groupby('temp_idx').agg(agg_map).reset_index()
        
        # 3. Join aggregated attributes back to merged_gdf
        # Use pandas merge but ensure we keep it as a GeoDataFrame
        merged_gdf = merged_gdf.merge(attr_data, on='temp_idx', how='left')
        
        # Cleanup
        merged_gdf.drop(columns=['temp_idx'], inplace=True)
        if 'index_right' in merged_gdf.columns:
            merged_gdf.drop(columns=['index_right'], inplace=True)
            
        print(f"Attributes preserved. Columns: {merged_gdf.columns.tolist()}")
    else:
        print("No level/height columns found in input to preserve.")

    # 4. Final Explosion (CRITICAL for Count Consistency)
    # The negative buffer might have split complex polygons into multiple parts.
    # We must explode them now so that 'Buildings After Merge' count is accurate.
    print("Ensuring result is fully exploded (resolving MultiPolygons)...")
    merged_gdf = merged_gdf.explode(index_parts=False)

    # Project back to original CRS if we changed it
    if original_crs and original_crs.is_geographic:
        merged_gdf = merged_gdf.to_crs(original_crs)
    
    print(f"Final feature count: {len(merged_gdf)}")
    
    # Save to file
    # Determine driver based on extension
    out_ext = os.path.splitext(output_path)[1].lower()
    driver = "GeoJSON"
    if out_ext == ".gpkg":
        driver = "GPKG"
    elif out_ext == ".shp":
        driver = "ESRI Shapefile"

    print(f"Saving merged footprints to {output_path} (Driver: {driver})...")
    merged_gdf.to_file(output_path, driver=driver)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Merge contiguous building footprints.")
    parser.add_argument("input_file", help="Path to input polygon file")
    parser.add_argument("output_file", help="Path to output merged file")
    parser.add_argument("--buffer", type=float, default=0.01, help="Buffer distance for merging (default: 0.01)")
    parser.add_argument("--layer", help="Layer name to read from multi-layer files (e.g. GPKG)")
    parser.add_argument("--levels_col", help="Name of the attribute column containing building level data (e.g. 'building:levels', 'num_floors')")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    merge_contiguous_polygons(args.input_file, args.output_file, args.buffer, args.layer, args.levels_col)

if __name__ == "__main__":
    main()
