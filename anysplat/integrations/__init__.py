"""
Integration helpers — bridge between the standalone GaussianSceneRestorer
and the AnySplat encoder pipeline.

These modules depend on AnySplat's source code being present and are provided
as reference implementations showing how to integrate the restorer without
modifying AnySplat source files directly.
"""
from .anysplat_composer import AnySplatWithRestorer
