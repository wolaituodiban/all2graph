def json_round(inputs, n):
    if isinstance(inputs, list):
        return [json_round(x, n) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: json_round(v, n) for k, v in inputs.items()}
    return round(float(inputs), n)
