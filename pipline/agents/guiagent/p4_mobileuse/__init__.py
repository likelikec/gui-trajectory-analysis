"""P4 hierarchical reflection configuration backed by the bundled MobileUse source."""
from study.modes.mobileuse_factory import build_mobileuse_agent_config

def build_p4_preview(**kwargs):
    return build_mobileuse_agent_config(mode='p4',**kwargs)
