import os


OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")


def _openai_client_class():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for model calls. Install it with `pip install -r requirements.txt`.") from exc

    return OpenAI


def _load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(override=True)


def infer_model_provider(model_config):
    if model_config.model_provider != "auto":
        return model_config.model_provider

    model_name = model_config.model_name
    if model_name.startswith(OPENAI_MODEL_PREFIXES):
        return "openai"
    if model_name.split('/')[0] == 'azure':
        return "openai"
    return "local"


def uses_hosted_openai(model_config):
    return infer_model_provider(model_config) == "openai"


def return_model(model_config):
    provider = infer_model_provider(model_config)
    OpenAI = _openai_client_class()

    if provider == "openai":
        _load_env()
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("OPENAI_API_KEY must be set when using --model_provider openai or an OpenAI model name.")

        client_kwargs = {"api_key": api_key}
        if model_config.openai_base_url is not None:
            client_kwargs["base_url"] = model_config.openai_base_url

        return OpenAI(**client_kwargs)

    if provider == "local":
        return OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{model_config.port_num}/v1"
        )

    raise ValueError(f"Unknown model provider: {provider}")
