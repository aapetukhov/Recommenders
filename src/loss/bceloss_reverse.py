from src.loss.bceloss import LogitsBCELoss


class LogitsBCELossReverse(LogitsBCELoss):
    """
    Wrapper loss for kt->dt direction. Reuses BCEWithLogitsLoss but
    exposes a dedicated Hydra target so the reverse pipeline stays
    isolated from the dt->kt one.
    """

    pass
