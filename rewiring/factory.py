from typing import Dict, Type, Optional

from rewiring.base import RewireStrategy
from rewiring.expander import EGPRewiring, CGPRewiring
from rewiring.fully_adjacent import FullyAdjacentRewiring
from rewiring.fosr import FosrRewiring
from rewiring.digl import DiglRewiring
from rewiring.sdrf import SdrfRewiring
from rewiring.borf import BorfRewiring
from rewiring.gtr import GtrRewiring

class RewireFactory:
    """
    Factory class for creating rewiring strategy instances.
    
    This factory provides a centralized way to instantiate different rewiring
    strategies based on their names. It maintains a registry of available
    strategies and can be extended with new strategies as needed.
    """
    
    # Registry of available rewiring strategies
    _strategies: Dict[str, Type[RewireStrategy]] = {
        'EGP': EGPRewiring,
        'CGP': CGPRewiring,
        'FullyAdjacent': FullyAdjacentRewiring,
        'FoSR': FosrRewiring,
        'DIGL': DiglRewiring,
        'SDRF': SdrfRewiring,
        'BORF': BorfRewiring,
        'GTR': GtrRewiring
    }

    @classmethod
    def get_strategy(cls, name: str) -> Optional[RewireStrategy]:
        """
        Get a rewiring strategy instance by name.
        
        Args:
            name: Name of the rewiring strategy
            
        Returns:
            Instance of the requested rewiring strategy, or None if not found
        """
        strategy_class = cls._strategies.get(name)
        if strategy_class:
            return strategy_class()
        return None
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[RewireStrategy]) -> None:
        """
        Register a new rewiring strategy.
        
        Args:
            name: Name for the rewiring strategy
            strategy_class: Class implementing the RewireStrategy interface
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        List the names of all registered rewiring strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())
