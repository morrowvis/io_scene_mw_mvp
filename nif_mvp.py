import re

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44

from . import nif_utils


def _normalize_prefix(filename, max_length=3):
    # lowercase a leading run of up to max_length letters ending at _ or space
    for n in range(1, max_length + 1):
        if len(filename) >= n + 1 and filename[n] in ('_', ' ') and filename[:n].isalpha():
            return filename[:n].lower() + filename[n:]
    return filename


def normalize_path(path):
    if not path or not isinstance(path, str):
        return path

    normalized = path.replace('/', '\\')
    if '\\' not in normalized:
        return normalized

    # the directory is lowercased, the filename is left as authored
    parts = normalized.rsplit('\\', 1)
    if len(parts) == 2:
        directory, filename = parts
        return directory.lower() + '\\' + filename

    return normalized


def sanitize_name(name, normalize_prefix=False, normalize_prefix_max_length=3):
    if not isinstance(name, str):
        name = str(name)

    name = re.sub(r'[\ud800-\udfff]', '', name)
    name = ''.join(c for c in name if c.isprintable())

    # separators and directory casing are always normalised, the prefix only on request
    name = normalize_path(name)
    if normalize_prefix:
        if '\\' in name:
            parts = name.rsplit('\\', 1)
            parts[1] = _normalize_prefix(parts[1], max_length=normalize_prefix_max_length)
            name = '\\'.join(parts)
        else:
            name = _normalize_prefix(name, max_length=normalize_prefix_max_length)

    return name or "Object"


def discard_detached_skins(data):
    """Strip skins that reference nodes outside the scene graph; those meshes
    import as static geometry rather than crashing."""
    in_scene = set()
    for root in data.roots:
        in_scene.add(id(root))
        if isinstance(root, nif.NiAVObject):
            in_scene.update(id(obj) for obj in root.descendants())

    for mesh in data.objects_of_type(nif.NiGeometry):
        skin = getattr(mesh, "skin", None)
        if not (skin and getattr(skin, "root", None) and getattr(skin, "bones", None)):
            continue
        detached = [n for n in (skin.root, *skin.bones) if id(n) not in in_scene]
        if detached:
            print(f"Warning: skin of '{mesh.name}' references nodes outside the scene "
                  f"(e.g. '{detached[0].name}'), importing as static")
            mesh.skin = None

def repair_scene(data):
    """Fix up live scene graphs from in-game exporters: repoint skin roots
    that are not ancestors of their bones, and bake BSMirroredNode's
    implicit reflection into the matrix."""
    # bake the implicit -1 point reflection; the file stores det=+1 matrices
    mirror = np.diag((-1.0, -1.0, -1.0, 1.0)).astype("<f")
    for obj in data.objects_of_type(nif.BSMirroredNode):
        if la.det(np.asarray(obj.matrix)[:3, :3]) > 0:
            obj.matrix = np.asarray(obj.matrix) @ mirror

    # build a parent map of the whole scene
    parents = {}
    for obj in data.objects_of_type(nif.NiNode):
        for child in obj.children:
            if child is not None:
                parents.setdefault(child, obj)

    def ancestry(o):  # root-first chain, including o itself
        chain = [o]
        while o in parents:
            o = parents[o]
            chain.append(o)
        chain.reverse()
        return chain

    for mesh in data.objects_of_type(nif.NiGeometry):
        skin = getattr(mesh, "skin", None)
        if not (skin and getattr(skin, "root", None) and getattr(skin, "bones", None)):
            continue

        # a live capture can bind skin.root to a same-named node elsewhere
        # in the graph; the matrix is authored against the mesh's own
        mesh_chain = ancestry(mesh)
        if not any(n is skin.root for n in mesh_chain):
            twin = next((n for n in reversed(mesh_chain)
                         if n is not mesh and n.name == skin.root.name), None)
            if twin is not None:
                print(f"Repointed skin root of '{mesh.name}' to its own "
                      f"ancestor '{twin.name}'")
                skin.root = twin

        chains = [ancestry(skin.root)] + [ancestry(b) for b in skin.bones]
        if all(skin.root in chain for chain in chains):
            continue  # already a common ancestor

        # nearest common ancestor of the old root and all bones
        new_root = None
        for level in zip(*chains):
            if all(n is level[0] for n in level) and isinstance(level[0], nif.NiNode):
                new_root = level[0]
            else:
                break

        if (new_root is None) or (new_root is skin.root):
            print(f"Warning: could not repair skin root of '{mesh.name}'")
            continue

        # keep semantics: root_to_skin_new = root_to_skin_old @ inv(old_root relative to new_root)
        offset = skin.root.matrix_relative_to(new_root)
        skin.data.matrix = np.asarray(skin.data.matrix) @ la.inv(offset)
        print(f"Repaired skin root of '{mesh.name}': '{skin.root.name}' -> '{new_root.name}'")
        skin.root = new_root

def apply_hierarchy_scales(data):
    """Freeze non-root node scales into the hierarchy: Blender rest bones are
    orthonormal, so scale left inside an armature breaks the animation math.
    World transforms are unchanged; a no-op on vanilla assets."""
    import copy

    def embedded_scale(obj):
        return abs(la.det(np.asarray(obj.rotation, dtype=np.float64))) ** (1.0 / 3.0)

    def is_uniform(obj):
        r = np.asarray(obj.rotation, dtype=np.float64)
        axes = la.norm(r, axis=0)
        return float(axes.max() - axes.min()) < 1e-3

    def is_animated(obj):
        if not hasattr(obj, "controllers"):
            return False
        kf = obj.controllers.find_type(nif.NiKeyframeController)
        if not (kf and kf.data):
            return False
        d = kf.data
        return bool(d.rotations.euler_data or len(d.rotations.keys)
                    or len(d.translations.keys) or len(d.scales.keys))

    subtree_animated_cache = {}

    def subtree_animated(obj):
        key = id(obj)
        cached = subtree_animated_cache.get(key)
        if cached is not None:
            return cached
        result = is_animated(obj)
        if not result and isinstance(obj, nif.NiNode):
            result = any(
                subtree_animated(c) for c in obj.children
                if c is not None and isinstance(c, nif.NiAVObject)
            )
        subtree_animated_cache[key] = result
        return result

    def freezable(obj):
        # negative scale is a stored reflection (BSMirroredNode), not a
        # scale; freezing consumes it and loses the mirror
        if obj.scale < 0:
            return False
        # only uniform scale inside animated content
        if abs(obj.scale * embedded_scale(obj) - 1.0) <= 1e-4:
            return False
        if not is_uniform(obj):
            if subtree_animated(obj):
                print(f"Warning: non-uniform scale on animated node '{obj.name}' cannot be frozen")
            return False
        if subtree_animated(obj):
            return True
        # a skinned mesh's own scale leaks into the bind poses; freeze it too
        skin = getattr(obj, "skin", None)
        if skin and getattr(skin, "bones", None):
            if any(b is not None and subtree_animated(b) for b in skin.bones):
                return True
        return False

    roots = [root for root in data.roots if isinstance(root, nif.NiNode)]
    if not any(
        freezable(obj)
        for root in roots
        for obj in root.descendants()
        if isinstance(obj, nif.NiAVObject)
    ):
        return

    seen_data = {}  # id(geometry data) -> factor it was baked with
    factors = {}    # id(node) -> accumulated scale frozen into it

    def freeze(node, inherited):
        if abs(inherited - 1.0) > 1e-6:
            node.translation = node.translation * inherited
            if node.bounding_volume:
                node.bounding_volume.apply_scale(inherited)
            for controller in node.controllers:
                if isinstance(controller, nif.NiKeyframeController) and controller.data:
                    controller.data.translations.apply_scale(inherited)

        frozen = freezable(node)
        own = 1.0
        if frozen:
            # normalize scale hidden in the rotation matrix
            embed = embedded_scale(node)
            if abs(embed - 1.0) > 1e-4:
                node.rotation = node.rotation / embed

            own = node.scale * embed
            kf = node.controllers.find_type(nif.NiKeyframeController)
            kfd = kf.data if (kf and kf.data) else None
            if kfd is not None and len(kfd.scales.keys):
                svals = kfd.scales.keys[:, 1]
                if float(svals.max() - svals.min()) < 1e-3:
                    # constant scale keys: fold into the freeze, apply once
                    s_anim = float(svals[0])
                    kfd.scales.keys = kfd.scales.keys[:0]
                    if abs(s_anim - own) > 1e-3:
                        print(f"Warning: '{node.name}' static scale {own:.3f} != animated scale {s_anim:.3f}; using animated")
                    own = node.scale * s_anim
                elif abs(embed - 1.0) > 1e-4:
                    # variable scale animation cannot be frozen; keep the
                    # keys and only report the (unfixable) embedded part
                    print(f"Warning: '{node.name}' has variable scale animation; import may be inexact")
            elif abs(embed - 1.0) > 1e-4:
                rot = kfd.rotations if kfd is not None else None
                if rot is not None and (rot.euler_data or len(rot.keys)):
                    # animated rotation overwrites it, so it never renders
                    print(f"Discarding junk rotation scale {embed:.3f} on animated node '{node.name}'")
                    own = node.scale
                else:
                    print(f"Freezing rotation scale {embed:.3f} of static node '{node.name}'")

        # total = scale removed from this node's frame by the freeze;
        # un-frozen nodes keep their own scale (only inherited applies)
        total = inherited * own
        factors[id(node)] = total
        if isinstance(node, nif.NiNode):
            if frozen:
                node.scale = 1.0
            for child in node.children:
                if child is not None and isinstance(child, nif.NiAVObject):
                    freeze(child, total)
        elif abs(total - 1.0) > 1e-6 and getattr(node, "data", None) is not None:
            # geometry leaf: bake the accumulated scale into vertices
            prior = seen_data.get(id(node.data))
            if prior is None:
                node.data.apply_scale(total)
                seen_data[id(node.data)] = total
            elif abs(prior - total) > 1e-6:
                # instanced data used at a different accumulated scale
                node.data = copy.deepcopy(node.data)
                node.data.apply_scale(total / prior)
                seen_data[id(node.data)] = total
            morpher = node.controllers.find_type(nif.NiGeomMorpherController)
            if morpher and morpher.data:
                morpher.data.apply_scale(total)
            if frozen:
                node.scale = 1.0

    for root in roots:
        print(f"Applying hierarchy scales under '{root.name}'")
        for child in root.children:
            if child is not None and isinstance(child, nif.NiAVObject):
                freeze(child, 1.0)

    # rescale skin binds to match the frozen bones, then normalize any
    # residual into the vertex data (bind poses must stay orthonormal)
    skin_seen = {}  # id(geometry data) -> residual factor baked
    for mesh in data.objects_of_type(nif.NiGeometry):
        skin = getattr(mesh, "skin", None)
        if not (skin and getattr(skin, "root", None)
                and getattr(skin, "bones", None) and getattr(skin, "data", None)):
            continue
        f_mesh = factors.get(id(mesh), 1.0)

        residual = None
        new_binds = []
        for bone, bone_data in zip(skin.bones, skin.data.bone_data):
            f_bone = factors.get(id(bone), 1.0)
            m = np.asarray(bone_data.matrix, dtype=np.float64).copy()
            m[:3, :] *= f_bone
            m[:3, :3] /= f_mesh
            r = abs(la.det(m[:3, :3])) ** (1.0 / 3.0)
            residual = r if residual is None else residual
            new_binds.append(m)

        if residual is None:
            continue
        if abs(residual - 1.0) > 1e-4:
            # fold the residual into the bind-space vertex data
            print(f"Normalizing residual bind scale {residual:.3f} of '{mesh.name}'")
            for m in new_binds:
                m[:3, :3] /= residual
            if getattr(mesh, "data", None) is not None:
                prior = skin_seen.get(id(mesh.data))
                if prior is None:
                    mesh.data.apply_scale(residual)
                    skin_seen[id(mesh.data)] = residual
                elif abs(prior - residual) > 1e-6:
                    mesh.data = copy.deepcopy(mesh.data)
                    mesh.data.apply_scale(residual / prior)
                    skin_seen[id(mesh.data)] = residual

        changed = abs(f_mesh - 1.0) > 1e-6 or abs(residual - 1.0) > 1e-4
        for bone, bone_data, m in zip(skin.bones, skin.data.bone_data, new_binds):
            if changed or abs(factors.get(id(bone), 1.0) - 1.0) > 1e-6:
                bone_data.matrix = m

        # root_to_skin relates skin-root space to mesh space; both may
        # have been rescaled by the freeze (crab: skin.root is 'Bip01'
        # itself, factor 3)
        f_root = factors.get(id(skin.root), 1.0)
        if abs(f_mesh - 1.0) > 1e-6 or abs(f_root - 1.0) > 1e-6:
            m = np.asarray(skin.data.matrix, dtype=np.float64).copy()
            m[:3, :] *= f_mesh
            m[:3, :3] /= f_root
            skin.data.matrix = m

def best_lod_children(lod_node, children):
    # keep the highest-detail level; lod_levels is (near, far) per child
    levels = lod_node.lod_levels
    # pair by ORIGINAL index: a null child still consumes a level slot
    paired = [(c, levels[i]) for i, c in enumerate(children)
              if c is not None and i < len(levels)]

    if len(paired) < 2:
        # Nothing to choose between, or level data too short to trust.
        present = [c for c in children if c is not None]
        return present[:1] if len(present) > 1 else children

    # Drop levels that can never render (far <= near), e.g. dr_mist_lava.nif.
    renderable = [p for p in paired if p[1][1] > p[1][0]] or paired

    # nearest level wins: smallest near, then smallest far to break ties
    best = min(renderable, key=lambda p: (p[1][0], p[1][1]))[0]
    return [best]

def apply_axis_corrections_one(root, bones, axis_correction):

    # aim each non-biped bone's Y at its children so sticks follow the limbs
    for node in bones:
        if "Bip01" in node.name:
            continue
        positions = [c.matrix_posed[:3, 3] for c in node.children if hasattr(c, "matrix_posed")]
        if not positions:
            continue
        direction = np.mean(positions, axis=0) - node.matrix_posed[:3, 3]
        length = la.norm(direction)
        if length < 1e-5:
            continue
        # child direction in the bone's local frame -> corrected Y column
        y = node.matrix_posed[:3, :3].T @ (direction / length)
        y = y / la.norm(y)
        # pick the remaining axes with minimal twist vs the default correction
        x = axis_correction[:3, 0].astype(np.float64)
        x = x - np.dot(x, y) * y
        if la.norm(x) < 1e-5:
            x = axis_correction[:3, 2].astype(np.float64)
            x = x - np.dot(x, y) * y
        x = x / la.norm(x)
        z = np.cross(x, y)
        correction = np.identity(4, dtype="<f")
        correction[:3, 0] = x
        correction[:3, 1] = y
        correction[:3, 2] = z
        node.axis_correction_override = correction

    # apply bone axis corrections
    for node in reversed(bones):
        node.matrix_posed = node.matrix_posed @ node.axis_correction
        node.matrix_local = node.matrix_local @ node.axis_correction
        for child in node.children:
            child.matrix_local = node.axis_correction_inverse @ child.matrix_local

    # apply anim axis corrections
    root_inverse = la.inv(root.matrix_world)
    for node in bones:
        kf_controller = node.source.controllers.find_type(nif.NiKeyframeController)
        if not (kf_controller and kf_controller.data):
            continue

        try:
            parent_matrix = node.parent.matrix_posed
            parent_matrix_uncorrected = parent_matrix @ node.parent.axis_correction_inverse
        except AttributeError:  # parent is not bone
            parent_matrix = node.parent.matrix_world if node.parent else ID44
            parent_matrix_uncorrected = parent_matrix

        matrix_world = parent_matrix @ node.matrix_local
        matrix_relative_to_root = root_inverse @ matrix_world

        posed_offset = la.solve(matrix_relative_to_root, root_inverse)
        posed_offset = posed_offset @ parent_matrix_uncorrected

        t = kf_controller.data.translations
        if len(t.values):
            rotation = posed_offset[:3, :3].T
            translation = posed_offset[:3, 3]
            # convert to pose space
            t.values[:] = t.values @ rotation + translation
            if t.key_type.name == "BEZ_KEY":
                t.in_tans[:] = t.in_tans @ rotation
                t.out_tans[:] = t.out_tans @ rotation

        r = kf_controller.data.rotations
        if r.euler_data:
            # euler keys expose no .values, so convert before correcting
            r.convert_to_quaternions()
        if len(r.values):
            # apply axis correction
            axis_fix = nif_utils.quaternion_from_matrix(node.axis_correction)
            r.values[:] = nif_utils.quaternion_mul(r.values, axis_fix)
            # convert to pose space; quaternion_from_matrix needs orthonormal
            rotation_only = posed_offset
            scale = la.norm(np.asarray(posed_offset)[:3, :3], axis=0)
            if not np.allclose(scale, 1.0, rtol=0, atol=1e-6):
                rotation_only = np.array(posed_offset, dtype=np.float64)
                rotation_only[:3, :3] /= scale
            to_posed = nif_utils.quaternion_from_matrix(rotation_only)
            r.values[:] = nif_utils.quaternion_mul(to_posed, r.values)
            # keep successive keys on one hemisphere, else Blender's
            # componentwise interpolation swings the long way (a snap)
            if len(r.values) > 1:
                dots = np.einsum("ij,ij->i", r.values[:-1], r.values[1:])
                flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
                r.values[1:] *= flips[:, None]
