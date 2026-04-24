from .base_causal_forcing import BaseModel, SelfForcingModel, TeacherForcingModel, BidirectionalModel
from .base_self_forcing import BaseModel as SFBaseModel, SelfForcingModel as SFSelfForcingModel
from .base_longlive import BaseModel as LLBaseModel, SelfForcingModel as LLSelfForcingModel


def get_base_classes(method='default'):
    """Return (BaseModel, SelfForcingModel) for the given method.

    Args:
        method: One of 'default' (Causal-Forcing/Self-Forcing base), 'self_forcing', 'longlive'

    Returns:
        Tuple of (BaseModel, SelfForcingModel) classes, plus method-specific classes
    """
    if method == 'default':
        from .base_causal_forcing import BaseModel, SelfForcingModel, TeacherForcingModel, BidirectionalModel
        return BaseModel, SelfForcingModel, TeacherForcingModel, BidirectionalModel
    elif method == 'self_forcing':
        from .base_self_forcing import BaseModel, SelfForcingModel
        return BaseModel, SelfForcingModel
    elif method == 'longlive':
        from .base_longlive import BaseModel, SelfForcingModel
        return BaseModel, SelfForcingModel
    else:
        raise ValueError(f"Unknown base method '{method}'. Choose from: 'default', 'self_forcing', 'longlive'")
