import bpy
import pathlib

# Define the folder path containing .nif files
nif_folder = r"C:/Users/<Username>/AppData/Local/ModOrganizer/Morrowind/overwrite/Export Cells"

# Get all .nif files in the folder
nif_files = pathlib.Path(nif_folder).glob("*.nif")

# Import each .nif file
for nif_file in nif_files:
    bpy.ops.import_scene.mw(
        filepath=str(nif_file),
        use_existing_materials=True,
        ignore_animations=False,
        ignore_armatures=False,
        authored_rest_pose=True,
        discard_root_transforms=True,
        preserve_root_scale=True,      # race scale rides in the root matrix
        ignore_billboard_nodes=True,
        ignore_collision_nodes=True,
        ignore_tri_shadow=True,
        ignore_nodes_under_switches="OFF, HARVESTED, Closed",
        filter_best_lod=True,
        use_texture_fallbacks=True,
    )
