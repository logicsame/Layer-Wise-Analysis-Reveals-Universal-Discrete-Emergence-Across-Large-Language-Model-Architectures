from typing import List

def calculate_patch_layers(num_layers: int, num_patches: int = 5) -> List[int]:
    """
    ✅ AUTOMATIC PATCHING LAYER SELECTION
    
    Intelligently select which layers to patch based on model size.
    Returns evenly-spaced layers covering the full depth.
    
    Args:
        num_layers: Total number of layers in the model
        num_patches: How many layers to patch (default: 5)
    
    Returns:
        List of layer indices to patch
    """
    if num_layers <= 10:
        # Very small model - patch every other layer
        return list(range(0, num_layers, 2))
    
    # Calculate spacing for even distribution
    # Always include: first layer (0), last layer (num_layers-1), and middle layers
    if num_patches >= num_layers:
        # Patch all layers
        return list(range(num_layers))
    
    # Evenly space the patches
    spacing = num_layers / (num_patches - 1)
    patch_layers = [round(spacing * i) for i in range(num_patches - 1)]
    patch_layers.append(num_layers - 1)  # Always include last layer
    
    # Remove duplicates and sort
    patch_layers = sorted(list(set(patch_layers)))
    
    print(f"📍 Auto-selected {len(patch_layers)} patch layers for {num_layers}-layer model: {patch_layers}")
    
    return patch_layers