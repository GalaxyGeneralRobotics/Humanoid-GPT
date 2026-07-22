import os
import cv2
import imageio
import numpy as np


def images_to_video(image_list, output_filename, fps=30, color_format="RGB", preset=None, threads=None):
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    frames = []
    for img in image_list:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if color_format.upper() == "BGR":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif color_format.upper() != "RGB":
            raise ValueError("color_format must be 'RGB' or 'BGR'")
        frames.append(arr)

    if not frames:
        raise ValueError("image_list is empty.")

    ext = os.path.splitext(output_filename)[1].lower()

    try:
        import imageio_ffmpeg

        kw = dict(fps=fps)
        if ext in {".mp4", ".m4v", ".mov"}:
            ffmpeg_params = ["-pix_fmt", "yuv420p"]
            if preset:
                ffmpeg_params += ["-preset", str(preset)]
            if threads:
                ffmpeg_params += ["-threads", str(threads)]
            kw.update(codec="libx264", ffmpeg_params=ffmpeg_params)

        with imageio.get_writer(output_filename, format="FFMPEG", **kw) as w:
            for f in frames:
                w.append_data(f)
        return
    except Exception as e_ff:
        print(f"[WARN] ffmpeg writer failed ({type(e_ff).__name__}: {e_ff}). Falling back to PyAV...")

    try:
        if ext in {".mp4", ".m4v", ".mov"}:
            for codec_try in ["h264", "mpeg4"]:
                try:
                    with imageio.get_writer(output_filename, fps=fps, codec=codec_try) as w:
                        for f in frames:
                            w.append_data(f)
                    return
                except Exception as _e:
                    print(f"[WARN] PyAV codec '{codec_try}' failed: {_e}")
            raise RuntimeError("No working codec found for mp4 (tried h264, mpeg4).")
        else:
            with imageio.get_writer(output_filename, fps=fps) as w:
                for f in frames:
                    w.append_data(f)
            return
    except Exception as e_av:
        raise RuntimeError(
            "Failed to write video with both ffmpeg and PyAV.\n"
            "Try: `pip install imageio-ffmpeg`, or use .avi/.mkv for testing;\n"
            "Also ensure the system FFmpeg encoder includes libx264 / h264 / mpeg4.\n"
            f"Root cause: {type(e_av).__name__}: {e_av}"
        )

