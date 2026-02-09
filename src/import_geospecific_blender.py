import bpy
import csv
import os
import math
import time
import sys

# --- CONFIGURATION ---
# Default paths if not provided
DEFAULT_CSV_PATH = r"e:\Carlos\GitHub\HybridGT\pipeline_output_hybrid\geospecific_instances.csv"
DEFAULT_MODELS_DIR = r"e:\Carlos\GitHub\HybridGT\pipeline_output_hybrid\geospecific_models"
COLLECTION_NAME = "Geospecific_Buildings"

def get_args():
    # Helper to parse args after "--"
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) >= 2:
            return args[0], args[1]
    return DEFAULT_CSV_PATH, DEFAULT_MODELS_DIR

def import_geospecific():
    csv_path, models_dir = get_args()
    
    print(f"Starting Import from {csv_path}...")
    print(f"Models Directory: {models_dir}")
    
    start_time = time.time()

    # 1. Create Collection
    if COLLECTION_NAME not in bpy.data.collections:
        city_coll = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(city_coll)
    else:
        city_coll = bpy.data.collections[COLLECTION_NAME]

    # 2. Read CSV Data
    instances = []
    unique_models = set()
    offset_x = 0
    offset_y = 0
    offset_z = 0
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            first = True
            for row in reader:
                if first:
                    offset_x = float(row['x'])
                    offset_y = float(row['y'])
                    offset_z = float(row['z'])
                    print(f"Setting Origin to First Building: ({offset_x}, {offset_y}, {offset_z})")
                    first = False
                    
                instances.append(row)
                unique_models.add(row['obj_filename'])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Found {len(instances)} instances.")

    # 3. Load and Place Models
    # Since almost every model is likely unique, we might skip the prototype logic if they are truly 1:1.
    # But keeping it robust: Load unique models, then instantiate.
    # Note: Loading thousands of OBJs is heavy.
    
    count = 0
    # Optimize: Don't load all at once if memory is concern.
    # But usually 16k simple meshes is fine for Blender if they are low poly.
    
    # We will import directly into the scene collection for now
    
    loaded_objects = {}
    
    for row in instances:
        obj_file = row['obj_filename']
        full_path = os.path.join(models_dir, obj_file)
        
        if not os.path.exists(full_path):
            if count % 100 == 0:
                print(f"Warning: Model not found: {full_path}")
            continue
            
        # Check if already loaded
        if obj_file in loaded_objects:
             # Instance from existing
             proto = loaded_objects[obj_file]
             obj = proto.copy()
             city_coll.objects.link(obj)
        else:
            # Import new
            # Deselect all
            bpy.ops.object.select_all(action='DESELECT')
            
            try:
                # Blender 4.0+
                bpy.ops.wm.obj_import(filepath=full_path, forward_axis='Y', up_axis='Z') 
                # Note: Trimesh exports Z-up by default usually? Or Y-up?
                # create_geospecific.py used trimesh.creation.extrude_polygon, which extrudes along Z.
                # So Z is up. Y is Forward?
                # Let's assume standard Z-Up.
            except AttributeError:
                # Blender < 4.0
                bpy.ops.import_scene.obj(filepath=full_path, axis_forward='Y', axis_up='Z')
                
            selected_objs = bpy.context.selected_objects
            if not selected_objs:
                continue
                
            obj = selected_objs[0] # Assume one object per file
            
            # Link to collection
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            city_coll.objects.link(obj)
            
            loaded_objects[obj_file] = obj
            
        # Set Transform
        x = float(row['x']) - offset_x
        y = float(row['y']) - offset_y
        z = float(row['z']) - offset_z
        # Geospecific rotation is usually 0 because geometry is already rotated in create_geospecific.py
        # But let's check angle_deg
        angle_deg = float(row['angle_deg'])
        
        obj.location = (x, y, z)
        
        if angle_deg != 0:
            obj.rotation_euler[2] = math.radians(angle_deg)
            
        count += 1
        if count % 100 == 0:
            print(f"Imported {count} / {len(instances)} buildings...")

    total_time = time.time() - start_time
    print(f"Finished! Imported {count} buildings in {total_time:.2f} seconds.")

if __name__ == "__main__":
    import_geospecific()
