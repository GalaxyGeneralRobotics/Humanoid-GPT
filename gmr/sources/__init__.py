"""Data sources for the three tracking modes.

Each source produces frame dicts of the form
``{human_body_name: (position[3], orientation_wxyz[4])}`` that feed directly
into ``GeneralMotionRetargeting.retarget(frame)``:

    - bvh        : load_lafan1_file(path, pns=...)  -> offline frame list
    - fbx_xsens  : XsensClient(...)                 -> real-time MVN stream
    - fbx_noitom : get_noitom_client()              -> real-time PNLink stream
                   (requires the external, hardware-specific ``noitom`` pkg)
"""

from .bvh import load_lafan1_file
from .xsens import XsensClient


def get_noitom_client(*args, **kwargs):
    """Lazily construct a Noitom PNLink client.

    ``NoitomClient`` lives in the external, platform-specific ``noitom``
    package (Linux on-board only). It is imported lazily so this package
    stays importable on machines without the Noitom SDK. The returned client
    exposes ``start_thread()`` / ``get_frame_data()`` / ``stop()`` and yields
    frame dicts compatible with ``src_human="fbx_noitom"``.
    """
    from noitom import NoitomClient
    return NoitomClient(*args, **kwargs)


__all__ = ["load_lafan1_file", "XsensClient", "get_noitom_client"]
