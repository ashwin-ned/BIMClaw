"""InputImages injected into the Jupyter kernel (adapted from SpatialClaw)."""

from typing import List, Optional

from PIL import Image

from bim_agent.kernel_types.frame_image import FrameImage


class InputImages(list):
    """List of input images for the BIM agent.

    InputImages[0] is always the BEV floor plan image.
    InputImages[1:] are the key interior walkthrough frames.

    Each element is a FrameImage carrying a frame_index.
    """

    def __init__(
        self,
        images: List,
        frame_indices: Optional[List[int]] = None,
        max_edge: Optional[int] = None,
    ):
        if not frame_indices:
            frame_indices = list(range(len(images)))
        wrapped = [
            img if isinstance(img, FrameImage)
            else FrameImage(img, idx, max_edge=max_edge)
            for img, idx in zip(images, frame_indices)
        ]
        super().__init__(wrapped)

    @property
    def frame_indices(self) -> List[int]:
        return [fi.frame_index for fi in self]

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            new = InputImages.__new__(InputImages)
            list.__init__(new, result)
            return new
        return result

    def __repr__(self) -> str:
        fi = self.frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        # Describe special role of frame 0
        prefix = "InputImages(floor_plan=frame_0, "
        return f"{prefix}{len(self)} images, frame_indices={fi_str})"
