import os

from openai.lib.azure import AsyncAzureOpenAI, AzureOpenAI

from r2r.base import LLMConfig


class OpenAILLMClientProvider():
    """A concrete class for creating OpenAI models."""

    def __init__(
            self,
            config: LLMConfig,
            *args,
            **kwargs,
    ) -> None:
        if not isinstance(config, LLMConfig):
            raise ValueError(
                "The provided config must be an instance of OpenAIConfig."
            )
        try:
            from openai import AsyncOpenAI, OpenAI  # noqa
        except ImportError:
            raise ImportError(
                "Error, `openai` is required to run an OpenAILLM. Please install it using `pip install openai`."
            )

        if config.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
                )

            self.async_client = AsyncOpenAI()
            self.client = OpenAI()

        elif config.provider == "openai_azure":
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                raise ValueError(
                    "OpenAI API key not found. Please set the AZURE_OPENAI_API_KEY environment variable."
                )
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                raise ValueError(
                    "OpenAI API key not found. Please set the AZURE_OPENAI_ENDPOINT environment variable."
                )
            self.async_client = AsyncAzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                                 api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION",
                                                                       "2024-02-15-preview"),
                                                 azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                                                 )
            self.client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                      api_version="2024-02-15-preview",
                                      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                                      )
        else:
            raise ValueError(
                "OpenAI LLM Client must be initialized with config with `openai` or `openai_azure` provider."
            )

    def get_async_client(self):
        return self.async_client

    def get_client(self):
        return self.client
