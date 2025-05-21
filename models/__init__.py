"""
Models package for VLN-CE implementation.
"""

from models.pointnetpp import PointNetPP

MODEL_REGISTRY = {
    "pointnetpp": PointNetPP,
}