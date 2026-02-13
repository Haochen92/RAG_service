from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types
from google import genai
import asyncio
import os
import time

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM") or "1536"


class RateLimitedGeminiEmbedding(GoogleGenAIEmbedding):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, *args, sleep_s=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_s = sleep_s

    async def aget_text_embedding_batch(self, texts, show_progress=True, **kwargs):
        embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            await asyncio.sleep(self.sleep_s)  # throttle per batch
            embeddings.extend(await self._aget_text_embeddings(batch))
        return embeddings


class GeminiTextLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        min_interval_s: float = 0.0,  # 👈 add this
    ):
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.min_interval_s = float(min_interval_s)
        self._last_call_ts = 0.0
        self._throttle_lock = asyncio.Lock()  # avoids race if concurrent calls

    async def _throttle(self):
        if self.min_interval_s <= 0:
            return
        async with self._throttle_lock:
            now = time.monotonic()
            wait = (self._last_call_ts + self.min_interval_s) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_ts = time.monotonic()

    async def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: list[str] | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        await self._throttle()

        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_count=n,
            stop_sequences=stop or None,
            response_mime_type=response_mime_type,
        )

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=cfg,
        )
        return (resp.text or "").strip()
