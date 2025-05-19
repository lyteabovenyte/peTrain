from models import MODEL_REGISTRY

def get_model(name, cfg):
    assert name in MODEL_REGISTRY, f"Unknown model: {name}"
    return MODEL_REGISTRY[name](cfg)