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
    if (descriptions["comp_type"] == "" and descriptions["material"] == ""):
        return None
    before_pos = " "
    if (descriptions["pos"] == "corner" or descriptions["pos"] == "edge" or descriptions["pos"] == "intersection"):
        before_pos = "on"
    statement = f"{descriptions['direction']} {descriptions['stepped']} crack{before_pos}{descriptions['pos']} {descriptions['material']} {descriptions['comp_type']} {descriptions['influence_range']}"
    return statement