import aiohttp
import argparse
import asyncio
import copy
import datetime
import os
import json
import math
import numpy as np
import random
import re
import requests
import secrets
from typing import List, Dict, Any
from uuid import uuid4
from loguru import logger
from time import sleep
import sys

OPENAI_API_BASE_URL = "https://api.openai.com"

class RateLimitError(RuntimeError):
    ...


class TooManyRequestsError(RuntimeError):
    ...


class BadResponseError(RuntimeError):
    ...


class TokensExhaustedError(RuntimeError):
    ...


class ContextLengthExceededError(RuntimeError):
    ...


class ServerOverloadedError(RuntimeError):
    ...


class ServerError(RuntimeError):
    ...

class OpenAIHandler:

    def __init__(self):
        """Constructor."""
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        self.used_tokens = 0
        self.max_tokens = None
        self.model = "gpt-4"
        self.openai_api_keys = []
        self.openai_api_keys.append(os.environ.get("OPENAI_API_KEY1"))
        self.openai_api_keys.append(os.environ.get("OPENAI_API_KEY2"))
        if not self.openai_api_keys:
            raise ValueError(
                "OPENAI_API_KEY environment variable or openai_api_key must be provided"
            )

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to OpenAI API.

        :param path: URL path to send request to.
        :type path: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        open_ai_key = random.choices(self.openai_api_keys, weights=(1, 1), k=1)[0]

        headers = {"Authorization": f"Bearer {open_ai_key}"}

        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_API_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=600.0,
            ) as result:
                if result.status != 200:
                    text = await result.text()
                    logger.error(f"OpenAI request error: {text}")
                    if "too many requests" in text.lower():
                        raise TooManyRequestsError(text)
                    if (
                        "rate limit reached" in text.lower()
                        or "rate_limit_exceeded" in text.lower()
                    ):
                        sleep(30)
                        raise RateLimitError(text)
                    elif "context_length_exceeded" in text.lower():
                        raise ContextLengthExceededError(text)
                    elif "server_error" in text and "overloaded" in text.lower():
                        raise ServerOverloadedError(text)
                    elif (
                        "bad gateway" in text.lower() or "server_error" in text.lower()
                    ):
                        raise ServerError(text)
                    else:
                        raise BadResponseError(text)
                result = await result.json()
                logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
                self.used_tokens += result["usage"]["total_tokens"]
                if self.max_tokens and self.used_tokens > self.max_tokens:
                    raise TokensExhaustedError(
                        f"Max token usage exceeded: {self.used_tokens}"
                    )
                logger.debug(f"token usage: {self.used_tokens}")
                return result

    async def _post_no_exc(self, *a, **k):
        """Post, ignoring all exceptions."""
        try:
            return await self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def generate_response(self, instruction: str, **kwargs) -> str:
        """Call OpenAI with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        messages = copy.deepcopy(kwargs.pop("messages", None) or [])
        model = kwargs.get("model", self.model)
        path = "/v1/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        payload["messages"] = messages
        if instruction:
            payload["messages"].append({"role": "user", "content": instruction})
        response = await self._post_no_exc(path, payload)
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return None
        text = response["choices"][0]["message"]["content"]

        return text

    async def gen_with_retry(self, prompt, messages=[], attempt=0, **api_params):
        result = await self.generate_response(
            prompt, messages=messages, **api_params
        )
        if result and result.strip():
            return result
        if attempt > 3:
            return None

        return await self.gen_with_retry(
            prompt, messages=messages, attempt=attempt + 1, **api_params
        )


# Less robust implementation.
# response = openai.chat.completions.create(
#     model="gpt-4",
#     messages=[
#             {"role": "system", "content": "You are a professional translator. You specialize in translating from English to Serbian."},
#             {"role": "user", "content": prompt},
#         ],
#     temperature=1.0,
#     top_p=1.0,
#     presence_penalty=0.0,
#     frequency_penalty=0.0
# )

# if not response.choices[0].finish_reason == 'stop':
#     raise Exception(f'Result is too long - retrying.')

# result = response.choices[0].message.content
# print(result)
