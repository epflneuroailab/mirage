"""MoviePy compatibility helpers.

Support both MoviePy 1.x, which exposes ``moviepy.editor.VideoFileClip``, and
MoviePy 2.x, which exposes ``moviepy.VideoFileClip`` at the top level.
"""



def VideoFileClip(*args, **kwargs):
    """Return a MoviePy ``VideoFileClip`` across supported versions."""
    try:
        from moviepy import VideoFileClip as _VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip as _VideoFileClip
    return _VideoFileClip(*args, **kwargs)
