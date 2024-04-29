import os
import importlib

MODEL_REGISTRY = {}
TOKENIZER_REGISTRY = {}


def ModelSelect(model_name_or_path):
    model = None
    for name in MODEL_REGISTRY.keys():
        if name in model_name_or_path.lower():
            model = MODEL_REGISTRY[name]
    if model is None:
        model = MODEL_REGISTRY['llama']
    return model


def TokenizerSelect(model_name_or_path):
    tokenizer_init = None
    for name in TOKENIZER_REGISTRY.keys():
        if name in model_name_or_path.lower():
            tokenizer_init = TOKENIZER_REGISTRY[name]
    if tokenizer_init is None:
        tokenizer_init = TOKENIZER_REGISTRY['llama']
    return tokenizer_init


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]

        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_tokenizer(name):
    def register_tokenizer_cls(cls):
        if name in TOKENIZER_REGISTRY:
            return TOKENIZER_REGISTRY[name]

        TOKENIZER_REGISTRY[name] = cls
        return cls

    return register_tokenizer_cls


def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)


# automatically import any Python files in the models/ directory
models_dir = os.path.join(os.path.dirname(__file__), 'language_model')
import_models(models_dir, "tinychart.model.language_model")
