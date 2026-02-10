"""
Unified Blender Import Script for HybridGT Pipeline
Imports both clustered (geotypical) and geospecific building models.

Usage:
  blender -P src/import_all_blender.py -- <output_dir> [max_geotypical] [max_geospecific] [terrain_geotiff]
  
  Arguments:
    output_dir      - Pipeline output directory
    max_geotypical  - Max geotypical buildings (default: 200, -1 = unlimited)
    max_geospecific - Max geospecific buildings (default: 200, -1 = unlimited)
    terrain_geotiff - Optional path to GeoTIFF terrain image
  
  Example:
    blender -P src/import_all_blender.py -- pipeline_output_fixed 200 100
    blender -P src/import_all_blender.py -- pipeline_output_fixed -1 -1 terrain.tif
"""

import bpy
import csv
import os
import math
import time
import sys

# --- CONFIGURATION ---
# Default paths (can be overridden via command line)
DEFAULT_OUTPUT_DIR = r"e:\Carlos\GitHub\HybridGT\pipeline_output"
DEFAULT_MAX_GEOTYPICAL = 200   # Limit for clustered buildings (-1 = unlimited)
DEFAULT_MAX_GEOSPECIFIC = 200  # Limit for geospecific buildings (-1 = unlimited)

# Angle offset to fix orientation (OBJ model width is along X, predicted angle is from X-axis)
# If buildings appear rotated 90° wrong, change this to 90 or -90
DEFAULT_ANGLE_OFFSET = 90  # degrees

# Collection Names
CLUSTERED_COLLECTION = "Clustered_Buildings"
GEOSPECIFIC_COLLECTION = "Geospecific_Buildings"
PROTOTYPES_COLLECTION = "Prototypes"
TERRAIN_COLLECTION = "Terrain"

def parse_args():
    """Parse command line args after '--'."""
    output_dir = DEFAULT_OUTPUT_DIR
    max_geotypical = DEFAULT_MAX_GEOTYPICAL
    max_geospecific = DEFAULT_MAX_GEOSPECIFIC
    terrain_geotiff = None
    angle_offset = DEFAULT_ANGLE_OFFSET
    
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) >= 1:
            output_dir = args[0]
        if len(args) >= 2:
            max_geotypical = int(args[1])
        if len(args) >= 3:
            max_geospecific = int(args[2])
        if len(args) >= 4:
            terrain_geotiff = args[3]
        if len(args) >= 5:
            angle_offset = float(args[4])
    
    return output_dir, max_geotypical, max_geospecific, terrain_geotiff, angle_offset

def ensure_collection(name, hide=False):
    """Create or get a collection by name."""
    if name not in bpy.data.collections:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
        if hide:
            coll.hide_viewport = True
            coll.hide_render = True
    else:
        coll = bpy.data.collections[name]
    return coll

def import_obj(filepath):
    """Import an OBJ file (Y-up), return the imported object.
    
    Clustered OBJ models are Y-up with -Z as forward direction.
    Blender uses Z-up with -Y as forward.
    """
    bpy.ops.object.select_all(action='DESELECT')
    try:
        # Blender 4.0+
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='NEGATIVE_Z', up_axis='Y')
    except AttributeError:
        # Blender < 4.0
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-Z', axis_up='Y')
    
    return bpy.context.selected_objects


def import_obj_zup(filepath):
    """Import an OBJ file (Z-up), return the imported object.
    
    Geospecific OBJ models are Z-up (created by trimesh extrude_polygon).
    No axis conversion needed - already matches Blender's coordinate system.
    """
    bpy.ops.object.select_all(action='DESELECT')
    try:
        # Blender 4.0+
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='NEGATIVE_Y', up_axis='Z')
    except AttributeError:
        # Blender < 4.0
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-Y', axis_up='Z')
    
    return bpy.context.selected_objects

def load_csv_instances(csv_path):
    """Load instances from a CSV file."""
    instances = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instances.append(row)
    except FileNotFoundError:
        print(f"Warning: CSV not found: {csv_path}")
    return instances

def import_clustered_buildings(output_dir, offset, proto_coll, city_coll, max_buildings=-1, angle_offset=0):
    """Import clustered (geotypical) buildings using Collection Instancing.
    
    Args:
        angle_offset: Degrees to add to predicted angle (default 0, try 90 if buildings appear misaligned)
    """
    csv_path = os.path.join(output_dir, "instances.csv")
    models_dir = os.path.join(output_dir, "obj_models")
    
    instances = load_csv_instances(csv_path)
    if not instances:
        print("No clustered instances found.")
        return 0
    
    # Apply limit if set (>= 0)
    if max_buildings >= 0 and len(instances) > max_buildings:
        print(f"Limiting geotypical buildings from {len(instances)} to {max_buildings}")
        instances = instances[:max_buildings]
        
    print(f"Loading {len(instances)} geotypical instances using collection instancing...")
    
    # Find unique models
    unique_models = sorted(set(row['obj_filename'] for row in instances))
    model_collections = {}
    
    # Create a sub-collection for each prototype model
    for obj_file in unique_models:
        full_path = os.path.join(models_dir, obj_file)
        if not os.path.exists(full_path):
            print(f"Warning: Model not found: {full_path}")
            continue
            
        imported = import_obj(full_path)
        if imported:
            obj = imported[0]
            obj.name = f"proto_{obj_file}"
            
            # Create a collection for this prototype
            proto_sub_coll = bpy.data.collections.new(f"Proto_{obj_file}")
            proto_coll.children.link(proto_sub_coll)
            
            # Move object to its prototype collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            proto_sub_coll.objects.link(obj)
            
            model_collections[obj_file] = proto_sub_coll
            
    print(f"Loaded {len(model_collections)} unique models as prototype collections.")
    
    # Create instanced empties for each building
    count = 0
    for row in instances:
        obj_file = row['obj_filename']
        if obj_file not in model_collections:
            continue
            
        x = float(row['x']) - offset[0]
        y = float(row['y']) - offset[1]
        z = float(row['z']) - offset[2]
        # Negate angle and add offset to fix coordinate system mismatch
        angle_rad = math.radians(-float(row['angle_deg']) + angle_offset)
        
        # Create empty as collection instance
        empty = bpy.data.objects.new(f"bldg_{count}", None)
        empty.instance_type = 'COLLECTION'
        empty.instance_collection = model_collections[obj_file]
        empty.location = (x, y, z)
        empty.rotation_euler = (0, 0, angle_rad)
        
        city_coll.objects.link(empty)
        count += 1
        
        if count % 1000 == 0:
            print(f"Created {count} collection instances...")
    
    print(f"Created {count} building instances.")
    return count

def import_geospecific_buildings(output_dir, offset, geo_coll, max_buildings=-1):
    """Import geospecific buildings (unique geometries)."""
    csv_path = os.path.join(output_dir, "geospecific_instances.csv")
    models_dir = os.path.join(output_dir, "geospecific_models")
    
    if not os.path.exists(csv_path):
        print("No geospecific instances CSV found.")
        return 0
        
    instances = load_csv_instances(csv_path)
    if not instances:
        print("No geospecific instances found.")
        return 0
    
    # Apply limit if set (>= 0)
    if max_buildings >= 0 and len(instances) > max_buildings:
        print(f"Limiting geospecific buildings from {len(instances)} to {max_buildings}")
        instances = instances[:max_buildings]
        
    print(f"Loading {len(instances)} geospecific instances...")
    
    count = 0
    for row in instances:
        obj_file = row['obj_filename']
        full_path = os.path.join(models_dir, obj_file)
        
        if not os.path.exists(full_path):
            continue
            
        imported = import_obj_zup(full_path)  # Geospecific uses Z-up (trimesh)
        if not imported:
            continue
            
        obj = imported[0]
        
        # Move to geospecific collection
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        geo_coll.objects.link(obj)
        
        # Position
        x = float(row['x']) - offset[0]
        y = float(row['y']) - offset[1]
        z = float(row['z']) - offset[2]
        angle_deg = float(row['angle_deg'])
        
        obj.location = (x, y, z)
        if angle_deg != 0:
            obj.rotation_euler[2] = math.radians(angle_deg)
            
        count += 1
        if count % 100 == 0:
            print(f"Imported {count} geospecific models...")
            
    return count


def import_terrain(geotiff_path, offset, terrain_coll):
    """Import a GeoTIFF as a textured plane for terrain."""
    if not os.path.exists(geotiff_path):
        print(f"Warning: Terrain file not found: {geotiff_path}")
        return None
    
    print(f"Loading terrain from: {geotiff_path}")
    
    # Try to read GeoTIFF metadata for dimensions and georeferencing
    try:
        # Try using rasterio if available (for precise georeferencing)
        import rasterio
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            width = bounds.right - bounds.left
            height = bounds.top - bounds.bottom
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            print(f"GeoTIFF bounds: {bounds}")
            print(f"GeoTIFF size: {width:.1f} x {height:.1f} meters")
    except ImportError:
        # Fallback: estimate from image dimensions (assume 1 pixel = 1 meter)
        print("Note: rasterio not available, using default terrain size")
        width = 2000  # Default 2km
        height = 2000
        center_x = offset[0]
        center_y = offset[1]
    
    # Create plane mesh
    import bmesh
    bm = bmesh.new()
    
    # Calculate corners relative to offset
    half_w = width / 2
    half_h = height / 2
    cx = center_x - offset[0]
    cy = center_y - offset[1]
    
    # Create 4 vertices for the plane
    v1 = bm.verts.new((cx - half_w, cy - half_h, 0))
    v2 = bm.verts.new((cx + half_w, cy - half_h, 0))
    v3 = bm.verts.new((cx + half_w, cy + half_h, 0))
    v4 = bm.verts.new((cx - half_w, cy + half_h, 0))
    
    # Create face
    bm.faces.new((v1, v2, v3, v4))
    
    # Add UV layer
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        for i, loop in enumerate(face.loops):
            if i == 0:
                loop[uv_layer].uv = (0, 0)
            elif i == 1:
                loop[uv_layer].uv = (1, 0)
            elif i == 2:
                loop[uv_layer].uv = (1, 1)
            else:
                loop[uv_layer].uv = (0, 1)
    
    # Create mesh and object
    mesh = bpy.data.meshes.new("TerrainMesh")
    bm.to_mesh(mesh)
    bm.free()
    
    terrain_obj = bpy.data.objects.new("Terrain", mesh)
    terrain_coll.objects.link(terrain_obj)
    
    # Create material with texture
    mat = bpy.data.materials.new(name="TerrainMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 0)
    
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)
    
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.location = (-300, 0)
    
    # Load image
    img = bpy.data.images.load(geotiff_path)
    tex_node.image = img
    
    # Connect nodes
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to object
    terrain_obj.data.materials.append(mat)
    
    print(f"Created terrain plane: {width:.0f}m x {height:.0f}m at center ({cx:.0f}, {cy:.0f})")
    return terrain_obj

def import_all():
    """Main function to import both clustered and geospecific buildings."""
    output_dir, max_geotypical, max_geospecific, terrain_geotiff, angle_offset = parse_args()
    print(f"\n=== HybridGT Blender Import ===")
    print(f"Output Directory: {output_dir}")
    print(f"Max Geotypical: {max_geotypical if max_geotypical >= 0 else 'unlimited'}")
    print(f"Max Geospecific: {max_geospecific if max_geospecific >= 0 else 'unlimited'}")
    print(f"Angle Offset: {angle_offset}°")
    if terrain_geotiff:
        print(f"Terrain: {terrain_geotiff}")
    
    start_time = time.time()
    
    # Determine global offset from first clustered instance
    csv_path = os.path.join(output_dir, "instances.csv")
    instances = load_csv_instances(csv_path)
    
    if instances:
        offset = (
            float(instances[0]['x']),
            float(instances[0]['y']),
            float(instances[0]['z'])
        )
        print(f"Global Offset (Origin): {offset}")
    else:
        # Fallback to geospecific
        geo_csv = os.path.join(output_dir, "geospecific_instances.csv")
        geo_instances = load_csv_instances(geo_csv)
        if geo_instances:
            offset = (
                float(geo_instances[0]['x']),
                float(geo_instances[0]['y']),
                float(geo_instances[0]['z'])
            )
        else:
            offset = (0, 0, 0)
        print(f"Global Offset (from geospecific): {offset}")
    
    # Create collections
    proto_coll = ensure_collection(PROTOTYPES_COLLECTION, hide=True)
    clustered_coll = ensure_collection(CLUSTERED_COLLECTION)
    geospecific_coll = ensure_collection(GEOSPECIFIC_COLLECTION)
    terrain_coll = ensure_collection(TERRAIN_COLLECTION)
    
    # Import terrain if specified
    if terrain_geotiff:
        import_terrain(terrain_geotiff, offset, terrain_coll)
    
    # Import both types (with limits and angle offset)
    clustered_count = import_clustered_buildings(output_dir, offset, proto_coll, clustered_coll, max_geotypical, angle_offset)
    geospecific_count = import_geospecific_buildings(output_dir, offset, geospecific_coll, max_geospecific)
    
    total_time = time.time() - start_time
    
    print(f"\n=== Import Complete ===")
    print(f"Clustered Buildings: {clustered_count}")
    print(f"Geospecific Buildings: {geospecific_count}")
    print(f"Total: {clustered_count + geospecific_count}")
    print(f"Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    import_all()
