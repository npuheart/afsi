"""
afsic.common — 公共工具模块

提供边界标记、速度边界条件、时间管理、日志等常用工具。
"""

from .utilities import (TimeManager, swanlab_init, swanlab_upload,
                         unique_filename, get_project_name, log, pressure_waveform)

from .boundaries import (tag_boundaries, rectangle_boundaries, box_boundaries,
                          MARKER_LEFT, MARKER_RIGHT, MARKER_BOTTOM, MARKER_TOP,
                          MARKER_FRONT, MARKER_BACK)

from .bcs import (UpVelocity2D, UpVelocity3D, TurekInlet, TurekInlet3D,
                   SinusoidalInlet, PipeInlet3D)
