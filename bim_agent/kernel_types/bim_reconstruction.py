"""BIMReconstruction kernel type: bridges bim_modalities → bim_agent kernel.

The agent receives this object as ``Reconstruction`` (same variable name as
SpatialClaw) so that geometry tools and BEV rendering work without changes.

This is a thin re-export wrapper.  The actual computation is in
bim_modalities.rgbd.BIMReconstruction.
"""

from __future__ import annotations

# Re-export so kernel code can do:
#   from bim_agent.kernel_types.bim_reconstruction import BIMReconstruction
from bim_modalities.rgbd import BIMReconstruction

__all__ = ["BIMReconstruction"]
