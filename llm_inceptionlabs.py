import llm
from pydantic import Field

from llm.default_plugins.openai_models import Chat, AsyncChat, SharedOptions

API_BASE = "https://api.inceptionlabs.ai/v1"

MODELS = (
    "mercury-3",
    "mercury-2.5",
    "mercury-2",
)


class InceptionOptions(SharedOptions):
    # Inception documents `max_completion_tokens` rather than `max_tokens`
    max_completion_tokens: int | None = Field(
        description="Maximum number of tokens to generate.",
        default=None,
    )
    # Inception uses four values, including `instant`, which the OpenAI
    # models in LLM's default plugin do not offer
    reasoning_effort: str | None = Field(
        description=(
            "How much the model should deliberate before answering. "
            "One of instant, low, medium or high."
        ),
        default=None,
    )
    # Inception-only: stream intermediate denoising steps instead of
    # just final text. Each chunk contains the full text as it is
    # refined, so chunks are not deltas of each other.
    diffusing: bool | None = Field(
        description=(
            "Stream intermediate denoising steps. Each chunk is the full "
            "text as it is refined, not a delta."
        ),
        default=None,
    )


class _InceptionShared:
    def __init__(self, model_id):
        # Prefix the LLM-facing model_id with our namespace so these
        # models cannot collide with other plugins (e.g. OpenRouter
        # serves the same upstream models). The raw provider ID is kept
        # in model_name, which is what gets sent to the API.
        super().__init__(
            model_id=f"inceptionlabs/{model_id}",
            model_name=model_id,
            api_base=API_BASE,
            supports_schema=True,
            supports_tools=True,
        )

    def __str__(self):
        return "Inception: {}".format(self.model_id)


class InceptionChat(_InceptionShared, Chat):
    needs_key = "inception"
    key_env_var = "INCEPTION_API_KEY"
    Options = InceptionOptions


class InceptionAsyncChat(_InceptionShared, AsyncChat):
    needs_key = "inception"
    key_env_var = "INCEPTION_API_KEY"
    Options = InceptionOptions


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "inception", InceptionChat.key_env_var)
    if not key:
        return
    for model_id in MODELS:
        register(InceptionChat(model_id), InceptionAsyncChat(model_id))
