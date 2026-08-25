import jax
from flax import struct

import re
from typing import Tuple, Any, Dict

@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: Any                                   
    obs: jax.Array                                   
    reward: jax.Array                               
    done: jax.Array                                 
    metrics: Dict[str, jax.Array]                   
    info: Dict[str, Any]                            
