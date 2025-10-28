"""Lightweight LLM wrapper for agents.

This module implements a resilient LLM client used by the DiabetesEye agents.
It attempts the following, in order:

- Google Generative Language SDK (google.ai.generativelanguage.TextServiceClient)
- Newer Google SDK surface (google.generativeai) if available
- REST call to Google's Generative Language endpoint using an API key
- OpenAI SDK (openai) as a fallback if OPENAI_API_KEY is present
- A deterministic local stub (offline/demo friendly)

All network attempts are logged to `llm_debug.log` next to this file for
diagnostics. The wrapper is conservative: failures are caught and the stub is
used so the agent pipeline remains runnable during development.
"""

from typing import Optional, Any
import os
import traceback
import json
from datetime import datetime

from crewai.llms.base_llm import BaseLLM


def _now():
    return datetime.utcnow().isoformat() + "Z"


def _stub_completion(prompt: str, role: Optional[str] = None) -> str:
    p = prompt.lower()
    if "explain" in p or "summary" in p:
        return "Imagem processada. Achados compatíveis com retinopatia diabética moderada; recomenda-se acompanhamento e controle glicêmico."
    if "plan" in p or "recommend" in p or "follow-up" in p:
        return "Recomenda-se retorno em 3 meses para avaliação, otimização de HBA1C e encaminhamento a oftalmologia se houver piora."
    return "Resumo: imagem analisada com sucesso; ver relatório detalhado."


class LLMClient:
    def __init__(self):
        self._client = None
        self._client_type = "stub"
        self.model = os.environ.get("GEMINI_MODEL", os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
        # Optional override to force provider preference: 'openai', 'google', or 'local'
        self.preferred = os.environ.get("LLM_PREFERRED", "").lower()
        self.debug_log = os.path.join(os.path.dirname(__file__), "llm_debug.log")
        if self.preferred:
            self._log(f"LLM preferred provider set to '{self.preferred}' via LLM_PREFERRED")

    def _log(self, *parts: str):
        try:
            with open(self.debug_log, "a", encoding="utf-8") as fh:
                fh.write(f"[{_now()}] ")
                for p in parts:
                    fh.write(p)
                fh.write("\n")
        except Exception:
            # Best-effort logging only
            pass

    def _try_init_google_sdk(self):
        """Try to initialize google.ai.generativelanguage TextServiceClient."""
        try:
            from google.ai.generativelanguage import TextServiceClient
            from google.api_core.client_options import ClientOptions

            client_options = ClientOptions(api_endpoint="generativelanguage.googleapis.com")
            client = TextServiceClient(client_options=client_options)
            self._client = client
            self._client_type = "google_ai_generativelanguage"
            self._log("Initialized google.ai.generativelanguage TextServiceClient")
            return True
        except Exception:
            self._log("google.ai.generativelanguage init failed:", traceback.format_exc())
            return False

    def _try_init_google_generativeai(self):
        """Try to initialize newer google.generativeai surface if present."""
        try:
            import google.generativeai as genai  # type: ignore
            # The newer SDK is typically used via functional API (configure + generate)
            self._client = genai
            self._client_type = "google_generativeai"
            self._log("Initialized google.generativeai module")
            return True
        except Exception:
            self._log("google.generativeai init failed:", traceback.format_exc())
            return False

    def _try_init_openai(self):
        try:
            import openai  # type: ignore
            self._client = openai
            self._client_type = "openai"
            self._log("Initialized openai SDK")
            return True
        except Exception:
            self._log("openai init failed:", traceback.format_exc())
            return False

    def _ensure_initialized(self):
        # If already initialized to a real client, return
        if self._client_type != "stub":
            return
        # Respect preferred provider ordering if requested
        pref = self.preferred
        if pref == "openai":
            if os.environ.get("OPENAI_API_KEY") and self._try_init_openai():
                return
            # fallthrough to other attempts
            if self._try_init_google_sdk():
                return
            if self._try_init_google_generativeai():
                return
        elif pref == "google":
            if self._try_init_google_sdk():
                return
            if self._try_init_google_generativeai():
                return
            if os.environ.get("OPENAI_API_KEY") and self._try_init_openai():
                return
        else:
            # Default order: Google SDKs -> OpenAI (only if explicitly allowed)
            if self._try_init_google_sdk():
                return
            if self._try_init_google_generativeai():
                return
            allow_openai = os.environ.get("LLM_ALLOW_OPENAI", "false").lower() in ("1", "true", "yes")
            if allow_openai and os.environ.get("OPENAI_API_KEY"):
                self._log("LLM_ALLOW_OPENAI is set; attempting OpenAI init")
                self._try_init_openai()
            else:
                self._log("Skipping OpenAI initialization (LLM_ALLOW_OPENAI not set or OPENAI_API_KEY missing)")

    def generate(self, prompt: str, role: Optional[str] = None, max_tokens: int = 256) -> str:
        # Ensure we attempted to init any available SDKs
        try:
            self._ensure_initialized()
        except Exception:
            self._log("_ensure_initialized error:", traceback.format_exc())

        # 1) If google.ai.generativelanguage client is available
        if self._client_type == "google_ai_generativelanguage":
            try:
                # Try expected call shape
                resp = self._client.generate_text(request={"model": self.model, "prompt": {"text": prompt}, "max_output_tokens": max_tokens})
                # resp may have candidates or text
                if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                    return getattr(resp.candidates[0], "content", str(resp.candidates[0]))
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
            except Exception:
                self._log("google_ai_generativelanguage generate_text failed:", traceback.format_exc())

        # 2) If google.generativeai module is available (functional API)
        if self._client_type == "google_generativeai":
            try:
                genai = self._client
                # configure via env key if present
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                # Use correct API: GenerativeModel
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                self._log("google.generativeai usage failed:", traceback.format_exc())

        # 3) REST fallback to Google Generative Language
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                import requests

                # Try multiple REST endpoint shapes to maximize compatibility across
                # API versions and model naming schemes. We try in order and log all
                # responses for diagnosis.
                rest_attempts = [
                    (f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}, "bearer"),
                    (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens}}, "key"),
                ]

                for url, body, auth_type in rest_attempts:
                    # Try different auth methods based on auth_type
                    attempts = []
                    if auth_type == "bearer":
                        attempts.append({"url": url, "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}})
                    elif auth_type == "key":
                        # 1) query param (existing approach)
                        attempts.append({"url": url + f"?key={api_key}", "headers": {"Content-Type": "application/json"}})
                        # 2) header with x-goog-api-key
                        attempts.append({"url": url, "headers": {"x-goog-api-key": api_key, "Content-Type": "application/json"}})
                        # 3) header with bearer token (some proxies accept it)
                        attempts.append({"url": url, "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}})

                    for attempt in attempts:
                        try:
                            r = requests.post(attempt["url"], json=body, timeout=15, headers=attempt["headers"])
                        except Exception:
                            self._log(f"Google REST request exception for url={attempt['url']} auth_type={auth_type}:", traceback.format_exc())
                            continue

                        # Log status and short body for debugging
                        self._log(f"Google REST attempt url={attempt['url']} auth_type={auth_type} status={r.status_code}")
                        txt = (r.text[:4000] + "...") if r.text and len(r.text) > 4000 else r.text
                        self._log(f"Google REST response snippet: {txt}")

                        if r.status_code >= 400:
                            # try next endpoint/form
                            continue

                        # parse and try to extract text in known shapes
                        try:
                            j = r.json()
                        except Exception:
                            return r.text

                        if isinstance(j, dict):
                            # OpenAI-compatible shape
                            if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
                                choice = j["choices"][0]
                                if isinstance(choice, dict) and "message" in choice:
                                    message = choice["message"]
                                    if isinstance(message, dict) and "content" in message:
                                        return message["content"]
                            # New Gemini API shape
                            if "candidates" in j and isinstance(j["candidates"], list) and len(j["candidates"]) > 0:
                                cand = j["candidates"][0]
                                if isinstance(cand, dict) and "content" in cand:
                                    content = cand["content"]
                                    if isinstance(content, dict) and "parts" in content and len(content["parts"]) > 0:
                                        part = content["parts"][0]
                                        if isinstance(part, dict) and "text" in part:
                                            return part["text"]
                            # if none matched, return full JSON as string
                            return str(j)
                # if all rest attempts failed, fall through to next fallback
            except Exception:
                self._log("Google REST request exception:", traceback.format_exc())

        # 4) OpenAI SDK fallback
        if self._client_type == "openai":
            try:
                openai = self._client
                # Use chat completions if available
                try:
                    resp = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.2)
                    if resp and "choices" in resp and len(resp["choices"]) > 0:
                        return resp["choices"][0]["message"]["content"]
                except Exception:
                    # Try legacy completion
                    resp = openai.Completion.create(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0.2)
                    if resp and "choices" in resp and len(resp["choices"]) > 0:
                        return resp["choices"][0].get("text", "")
            except Exception:
                self._log("openai invocation failed:", traceback.format_exc())

        # Last resort: deterministic local stub
        self._log("Falling back to local stub completion")
        return _stub_completion(prompt, role=role)


# Module-level default client
DEFAULT_LLM = LLMClient()


class GeminiFlashLLM(BaseLLM):
    """CrewAI BaseLLM adapter that reuses the resilient Gemini client."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float | None = 0.2,
        stop: list[str] | None = None,
        max_tokens: int = 512,
        client: LLMClient | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, stop=stop)
        self.max_tokens = max_tokens
        self._client = client or LLMClient()
        self._client.model = model

    def call(  # type: ignore[override]
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        prompt = self._format_messages(messages)
        return self._client.generate(prompt, role=getattr(from_agent, "role", None), max_tokens=self.max_tokens)

    def _format_messages(self, messages: str | list[dict[str, str]]) -> str:
        if isinstance(messages, str):
            return messages
        formatted = []
        for item in messages:
            role = item.get("role", "user").upper()
            content = item.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
