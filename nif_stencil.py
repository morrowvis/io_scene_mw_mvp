"""Stencil roles of a NIF's shapes, from NiStencilProperty and NiZBufferProperty
(both inherited down the scene graph) as the engine applies them. Ported from
Vvardenworks' src/world/stencil.rs.

The "hole" trick mods build on them: a mask shape marks its pixels (ALWAYS,
REPLACE), then a contents shape draws only there (EQUAL) with the depth test
off, over the wall in front of it.
"""

from es3 import nif

PLAIN = "plain"
MASK = "mask"          # writes the stencil without testing it: marks an opening
AREA = "area"          # NEVER passes but writes on fail: marks an area, draws nothing
CONTENTS = "contents"  # tests the stencil, depth test off: seen through the opening
CLIP = "clip"          # tests the stencil, depth test on: clipped to the opening

# NiStencilProperty's test functions and actions as stored.
TEST_NEVER = 0
TEST_ALWAYS = 7
ACTION_KEEP = 0
ACTION_INVERT = 5


def role(stencil=None, zbuffer=None):
    """The role of a shape drawn with these (inherited) properties."""
    # No stencil property, or a disabled one, tests ALWAYS and keeps.
    if stencil is None or not stencil.stencil_enabled:
        return PLAIN

    test = int(stencil.stencil_function)

    # The engine has no wrap actions; values past INVERT keep.
    def act(action):
        action = int(action)
        return action if 0 <= action <= ACTION_INVERT else ACTION_KEEP

    fail = act(stencil.fail_action)
    depth_fail = act(stencil.pass_z_fail_action)
    passed = act(stencil.pass_action)
    depth_test = zbuffer is None or bool(zbuffer.z_buffer_test)

    can_fail = test != TEST_ALWAYS
    can_pass = test != TEST_NEVER
    writes = ((can_fail and fail != ACTION_KEEP)
              or (can_pass and depth_test and depth_fail != ACTION_KEEP)
              or (can_pass and passed != ACTION_KEEP))

    if test == TEST_ALWAYS:
        return MASK if writes else PLAIN
    if test == TEST_NEVER:
        # Without a write NEVER does nothing. es3 also reads an out-of-range
        # function as NEVER where the engine draws it as ALWAYS: plain either way.
        return AREA if writes else PLAIN
    return CLIP if depth_test else CONTENTS


def classify(stream):
    """{shape: role} for every NiGeometry in the scene graph, as the engine
    inherits the properties: a node's own replace those from above."""
    roles = {}
    stack = [(root, None, None) for root in stream.roots if isinstance(root, nif.NiAVObject)]
    while stack:
        obj, stencil, zbuffer = stack.pop()
        for prop in obj.properties:
            if isinstance(prop, nif.NiStencilProperty):
                stencil = prop
            elif isinstance(prop, nif.NiZBufferProperty):
                zbuffer = prop
        if isinstance(obj, nif.NiGeometry):
            # a shape reached twice keeps its first path's role
            roles.setdefault(obj, role(stencil, zbuffer))
        for child in getattr(obj, "children", ()):
            if isinstance(child, nif.NiAVObject):
                stack.append((child, stencil, zbuffer))
    return roles
