import pytest
from llm.plugins import pm


@pytest.fixture
def registered():
    class Capture:
        def __init__(self):
            self.pairs = []

        def __call__(self, model, async_model=None, aliases=None):
            self.pairs.append((model, async_model, aliases))

    return Capture()


def test_plugin_is_installed(monkeypatch):
    # LLM skips entry-point loading under pytest (sys._called_from_test),
    # so load manually the same way load_plugins() does:
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    pm.load_setuptools_entrypoints("llm")
    try:
        names = [mod.__name__ for mod in pm.get_plugins()]
        assert "llm_inceptionlabs" in names
    finally:
        for name, mod in pm.list_name_plugin():
            if name == "llm_inceptionlabs":
                pm.unregister(mod)


def test_models_register_with_key(monkeypatch, registered):
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    from llm_inceptionlabs import InceptionChat, MODELS, register_models

    register_models(registered)
    assert [(m.model_id, am.model_id) for m, am, _ in registered.pairs] == [
        ("mercury-3", "mercury-3"),
        ("mercury-2.5", "mercury-2.5"),
        ("mercury-2", "mercury-2"),
    ]
    for model, async_model, _ in registered.pairs:
        assert isinstance(model, InceptionChat)
        # Each sync model registered in an (sync, async) pair
        assert async_model is not None and async_model.model_id == model.model_id


def test_no_registration_without_key(monkeypatch, registered):
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    from llm_inceptionlabs import register_models

    register_models(registered)
    assert registered.pairs == []


def test_options_accept_mercury_values():
    from llm_inceptionlabs import InceptionOptions

    options = InceptionOptions(
        reasoning_effort="instant", diffusing=True, max_completion_tokens=8192
    )
    assert options.reasoning_effort == "instant"
    assert options.diffusing is True
    assert options.max_completion_tokens == 8192


def test_model_str():
    from llm_inceptionlabs import InceptionChat

    assert str(InceptionChat("mercury-3")) == "Inception: mercury-3"
