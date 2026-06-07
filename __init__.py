"""
comfyui-gpt-image2-parallel
Async/parallel fork: OpenAI gpt-image-2 with your own key. Two parallel nodes.

Defensive: if the node fails to import, it logs and ComfyUI keeps starting.
"""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
try:
    from .nodes_api import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    import traceback
    print("[comfyui-gpt-image2-parallel] Failed to load node (ComfyUI keeps running):")
    traceback.print_exc()
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
