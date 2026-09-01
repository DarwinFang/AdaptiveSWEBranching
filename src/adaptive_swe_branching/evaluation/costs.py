from adaptive_swe_branching.data.records import Cost


def cost_value(cost: Cost, axis: str) -> float:
    if axis == "total_tokens":
        return float(cost.total_tokens)
    if not hasattr(cost, axis):
        raise ValueError(f"unknown cost axis: {axis}")
    return float(getattr(cost, axis))
