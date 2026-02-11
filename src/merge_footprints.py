import geopandas as gpd
import argparse
import os
import sys
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

def merge_contiguous_polygons(input_path, output_path, buffer_dist=0.01, layer=None):
    """
    Merges contiguous or overlapping polygons in a GeoJSON/Shapefile/GeoPackage.

    Args:
        input_path: Path to input building footprints.
        output_path: Path to save merged footprints.
        buffer_dist: Small buffer distance to ensure touching polygons intersect.
        layer: Layer name for GeoPackage input files.
    """
    print(f"Loading footprints from {input_path}...")
    read_kwargs = {}
    if layer:
        read_kwargs['layer'] = layer
    gdf = gpd.read_file(input_path, **read_kwargs)
    
    # Determine output driver from extension
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == '.gpkg':
        out_driver = 'GPKG'
    else:
        out_driver = 'GeoJSON'

    if len(gdf) == 0:
        print("Empty input file.")
        gdf.to_file(output_path, driver=out_driver)
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
    
    # Project back to original CRS if we changed it
    if original_crs and original_crs.is_geographic:
        merged_gdf = merged_gdf.to_crs(original_crs)
    
    print(f"Final feature count: {len(merged_gdf)}")
    
    # Save to file
    print(f"Saving merged footprints to {output_path}...")
    merged_gdf.to_file(output_path, driver=out_driver)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Merge contiguous building footprints.")
    parser.add_argument("input_file", help="Path to input polygon file (shp, geojson, gpkg)")
    parser.add_argument("output_file", help="Path to output merged file")
    parser.add_argument("--buffer", type=float, default=0.01, help="Buffer distance for merging (default: 0.01)")
    parser.add_argument("--layer", default=None, help="Layer name for GeoPackage (.gpkg) input files")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    merge_contiguous_polygons(args.input_file, args.output_file, args.buffer, layer=args.layer)

if __name__ == "__main__":
    main()
