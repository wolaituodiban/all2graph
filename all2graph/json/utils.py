def unstructurize_dict(d):
    output = {}
    for k, v in d.items():
        temp = output
        k_split =  k.split('_')
        for kk in k_split[1:-1]:
            if kk not in temp:
                temp[kk] = {}
            temp = temp[kk]
        temp[k_split[-1]] = v
    return output


def get_json_depth(obj, depth=0):
    if isinstance(obj, list):
        if len(obj) > 0:
            return max(get_json_depth(item, depth+1) for item in obj)
        else:
            return depth
    elif isinstance(obj, dict):
        if len(obj) > 0:
            return max(get_json_depth(item, depth+1) for item in obj.values())
        else:
            return depth
    else:
        return depth
