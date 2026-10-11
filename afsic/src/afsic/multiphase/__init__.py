"""Experimental serial 2D gas/liquid/elastic-solid backend.

Conservative MAC fluid + FEniCSx TRI3 solid. This is a first-order baseline,
not a reproduction of the high-order/AMR/TS-IFED implementation in IBAMR.
"""
from .grid import MACGrid
from .levelset import PhaseProperties
from .fluid import ConservativeMACSolver
from .coupling import MACCoupling
from .solid import ElasticSolid2D
from .simulation import ThreePhaseSimulation

__all__ = ["MACGrid", "PhaseProperties", "ConservativeMACSolver", "MACCoupling",
           "ElasticSolid2D", "ThreePhaseSimulation"]
