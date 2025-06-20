from typing import List, Dict

def format_creators(creators: List[Dict[str, str]], concise: bool = False) -> str:
    """
    Format creator names into a string.
    
    Args:
        creators: List of creator objects from Zotero.
        concise: If True, uses abbreviated format (e.g., "Smith et al." for multiple authors).
        
    Returns:
        Formatted string with creator names.
    """
    if not creators:
        return "No authors listed"
    
    if concise:
        # In concise mode, show first author and "et al." if multiple
        if len(creators) == 1:
            creator = creators[0]
            if "lastName" in creator:
                return creator["lastName"]
            elif "name" in creator:
                # For institutional authors, take first part
                return creator["name"].split(",")[0].split(" ")[0]
        else:
            # Multiple authors: "FirstAuthor et al."
            creator = creators[0]
            if "lastName" in creator:
                return f"{creator['lastName']} et al."
            elif "name" in creator:
                return f"{creator['name'].split(',')[0]} et al."
    
    # Full mode (existing behavior)
    names = []
    for creator in creators:
        if "firstName" in creator and "lastName" in creator:
            names.append(f"{creator['lastName']}, {creator['firstName']}")
        elif "name" in creator:
            names.append(creator["name"])
    return "; ".join(names) if names else "No authors listed"
