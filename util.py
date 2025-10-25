# this function combines the various description components into a single descriptive statement
# this tries to create a natural sentence which CLIP was trained on

def combine_descriptions(ws, row):
    descriptions = {
        "comp_type": "",
        "material": "",
        "pos": "",
        "stepped": "",
        "direction": "",
        "influence_range": "across "

    }
    for i in range(3, 9):
        key = list(descriptions.keys())[i - 3]
        value = ws.cell(row, i).value
        if value and str(value).lower() != "nan":
            descriptions[key] = str(value).strip().lower()
        else:
            descriptions[key] = ""
    if (descriptions["comp_type"] == "" or descriptions["material"] == ""):
        return None
    statement = f"{descriptions['stepped']} {descriptions['direction']} crack on {descriptions['material']} {descriptions['comp_type']}"
    # if (descriptions["pos"] == "corner" or descriptions["pos"] == "edge" or descriptions["pos"] == "intersection"):
        # before_pos = "on"
    # statement = f"{descriptions['direction']} {descriptions['stepped']} crack{before_pos}{descriptions['pos']} {descriptions['material']} {descriptions['comp_type']} " # {descriptions['influence_range']}
    return statement

def remove_duplicates(strings):
    seen = set()
    result = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result
