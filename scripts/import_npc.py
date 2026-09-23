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
        attach_keyframe_data=True,
        discard_root_transforms=True,
        preserve_root_scale=True,      # race scale rides in the root matrix
        use_existing_materials=True,
        ignore_collision_nodes=True,
        ignore_animations=False,
        ignore_armatures=False,
        ignore_billboard_nodes=True,
        ignore_tri_shadow=True,
        filter_best_lod=True,
        use_texture_fallbacks=True,
        extended_material_names=True,
        normalize_prefix=True,
        normalize_prefix_root=True,
        authored_rest_pose=True,
    )
