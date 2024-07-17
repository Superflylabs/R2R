"""A module for creating OpenAI model abstractions."""

import logging
import os
from typing import Union

from r2r.base import (
    LLMChatCompletion,
    LLMChatCompletionChunk,
    LLMConfig,
    LLMProvider,
)
from r2r.base.abstractions.llm import GenerationConfig
from r2r.providers.llms.openai.openai_client_provider import OpenAILLMClientProvider

logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
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
        super().__init__(config)
        self.config: LLMConfig = config

        client_provider = OpenAILLMClientProvider(config)

        self.async_client = client_provider.get_async_client()
        self.client = client_provider.get_client()

    def get_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletion:
        if generation_config.stream:
            raise ValueError(
                "Stream must be set to False to use the `get_completion` method."
            )
        return self._get_completion(messages, generation_config, **kwargs)

    def get_completion_stream(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletionChunk:
        if not generation_config.stream:
            raise ValueError(
                "Stream must be set to True to use the `get_completion_stream` method."
            )
        return self._get_completion(messages, generation_config, **kwargs)

    def _get_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> Union[LLMChatCompletion, LLMChatCompletionChunk]:
        """Get a completion from the OpenAI API based on the provided messages."""

        # Create a dictionary with the default arguments
        args = self._get_base_args(generation_config)

        args["messages"] = messages

        # Conditionally add the 'functions' argument if it's not None
        if generation_config.functions is not None:
            args["functions"] = generation_config.functions

        args = {**args, **kwargs}
        # Create the chat completion
        return self.client.chat.completions.create(**args)

    def _get_base_args(
        self,
        generation_config: GenerationConfig,
    ) -> dict:
        """Get the base arguments for the OpenAI API."""

        args = {
            "model": generation_config.model,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "stream": generation_config.stream,
            # TODO - We need to cap this to avoid potential errors when exceed max allowable context
            "max_tokens": generation_config.max_tokens_to_sample,
        }

        return args

    async def aget_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletion:
        if generation_config.stream:
            raise ValueError(
                "Stream must be set to False to use the `aget_completion` method."
            )
        return await self._aget_completion(
            messages, generation_config, **kwargs
        )

    async def _aget_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> Union[LLMChatCompletion, LLMChatCompletionChunk]:
        """Asynchronously get a completion from the OpenAI API based on the provided messages."""

        # Create a dictionary with the default arguments
        args = self._get_base_args(generation_config)

        args["messages"] = messages

        # Conditionally add the 'functions' argument if it's not None
        if generation_config.functions is not None:
            args["functions"] = generation_config.functions

        args = {**args, **kwargs}
        # Create the chat completion
        return await self.async_client.chat.completions.create(**args)
