import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
import os
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

def create_oriented_rectangle(cx, cy, width, height, angle_deg):
    """Creates a Shapely Polygon representing an oriented rectangle."""
    # Create an axis-aligned rectangle centered at the origin
    hw = width / 2.0
    hh = height / 2.0
    
    # Vertices of the rectangle
    # We assume 'width' is X and 'height' is Y in local space
    coords = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh),
        (-hw, -hh)
    ]
    
    rect = Polygon(coords)
    
    # Rotate (Shapely rotates CCW by default)
    # The models rotate around the center
    rotated_rect = rotate(rect, angle_deg, origin=(0, 0))
    
    # Translate to centroid
    final_rect = translate(rotated_rect, xoff=cx, yoff=cy)
    
    return final_rect

def main():
    parser = argparse.ArgumentParser(description="Reconstruct geotypical building footprints from instances CSV.")
    parser.add_argument("instances_csv", help="Path to input instances.csv")
    parser.add_argument("output_geojson", help="Path to save output GeoJSON")
    parser.add_argument("--crs", default="EPSG:32630", help="Coordinate Reference System (default: EPSG:32630)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.instances_csv):
        print(f"Error: File not found: {args.instances_csv}")
        return
        
    print(f"Loading instances from {args.instances_csv}...")
    df = pd.read_csv(args.instances_csv)
    
    if len(df) == 0:
        print("Empty CSV file.")
        return
        
    print(f"Reconstructing {len(df)} geotypical footprints...")
    
    geometries = []
    for idx, row in df.iterrows():
        # Using pred_width, pred_height and angle_deg
        # Note: In HybridGT, width is X and height is Y in the footrint
        # but the prediction script might have swapped them.
        # However, instances.csv preserves the orientation used during clustering.
        
        rect = create_oriented_rectangle(
            row['x'], 
            row['y'], 
            row['pred_width'], 
            row['pred_height'], 
            row['angle_deg']
        )
        geometries.append(rect)
        
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=args.crs)
    
    # Save to file
    print(f"Saving to {args.output_geojson}...")
    gdf.to_file(args.output_geojson, driver='GeoJSON')
    
    print("Done.")

if __name__ == "__main__":
    main()
