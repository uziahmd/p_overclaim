"""Gemini judges through the google-genai SDK (2.34, Interactions API), on two routes.

Routes (the route that served a call is recorded in the response, the judge record and the cost log):
- "aistudio": the Gemini Developer API with GEMINI_API_KEY, `genai.Client(enterprise=False, api_key=...)`.
  It is billed to the AI Studio prepaid balance; HTTP 402 means the balance is exhausted (BalanceExhausted: stop).
- "vertex": the Gemini Enterprise Agent Platform API (formerly Vertex AI). When GOOGLE_CLOUD_PROJECT is set it uses
  application default credentials, `genai.Client(enterprise=True, project=..., location="global")` (route label
  "vertex:adc"); otherwise the express-mode key VERTEX_API_KEY, `genai.Client(enterprise=True, api_key=...)`
  ("vertex:key"). It is billed to the Google Cloud project.

Both clients are built with explicit arguments. GOOGLE_API_KEY must not be set (the SDK would prefer it over
GEMINI_API_KEY), and the GOOGLE_GENAI_USE_* switches are never relied on.

Request: interactions.create(model, input, generation_config={thinking_level, max_output_tokens},
response_format={"type": "text", "mime_type": "application/json", "schema": ...} when a schema is given, store=False).
These models cannot turn thinking off; thought tokens are billed as output. No fallbacks.
Usage fields logged: total_input_tokens, total_output_tokens, total_thought_tokens, total_cached_tokens.

Pricing (Uzair's memo of 2026-11-08, section 1.3, for the rows moved to the aistudio route): list prices from the
price table; the cached input tokens (implicit caching, total_cached_tokens) at the table's implicit_cache_read price
when it has one (its own key, so the projections' cache_read pricing of the Gemini rows stays "as uncached input"),
the rest of the input at the input price. total_cached_tokens is taken as a subset of total_input_tokens
(the Gemini API's cachedContentTokenCount is a subset of promptTokenCount); should a response report more cached
than input tokens, all input is priced at the full rate and the cached tokens on top, the upper bound, with a
warning. Thought tokens are billed as output.

Stops (memo section 1.3): HTTP 402 (BalanceExhausted) and an HTTP 429 that is not a per-minute rate limit
(QuotaExhausted, a BalanceExhausted: a daily or project quota or a billing refusal) are raised without a retry; a
per-minute 429 is retried by the client's backoff. The rule (classify_429), on the error's JSON body (the
google.genai.errors APIError's `details`, or the Interactions client's compat error's `body`): when the body has
QuotaFailure violations with quota ids, it is a rate limit only if every quota id names a per-minute limit; a body
without quota ids is a rate limit when its message names a per-minute limit or its RetryInfo is at most
RETRY_INFO_MAX_S; anything else stops. The SDK's own retry policy is built without 429 (SDK_HTTP_OPTIONS), so the
first 429 reaches this rule unretried; the first 429 of a process is logged with its full body so the classification
can be checked afterwards. An HTTP 403 (the policy refusal of 2026-11-08) is neither retried nor converted; the
stages block the provider for the invocation on it (is_permission_denied).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

from overclaim.graders.cache import DiskCache
from overclaim.graders.judge import CostLogger, LLMResponse, _complete_cached, _credential, _payload, _with_backoff, estimate_tokens

logger = logging.getLogger(__name__)

ROUTES = ("aistudio", "vertex")
VERTEX_LOCATION = "global"
PER_MINUTE_MARKERS = ("perminute", "per minute", "per-minute")
RETRY_INFO_MAX_S = 120.0
# The SDK's own retry policy for every client built here: the google-genai Interactions client resends 408, 409, 429
# and 5XX up to three times by default (an uncapped Retry-After header replaces its delay). 429 is left out so that a
# quota or billing 429 is never resent and a per-minute 429 is retried by _with_backoff alone (memo section 1.3).
SDK_RETRY_STATUS_CODES = (408, 500, 502, 503, 504)
PROBE_PROMPT = "Reply with the single word OK."
PROBE_PURPOSE = "aistudio_probe"


class BalanceExhausted(RuntimeError):
    """HTTP 402 from the Gemini API: the prepaid balance is exhausted. Stop and report; never retry."""


class QuotaExhausted(BalanceExhausted):
    """HTTP 429 RESOURCE_EXHAUSTED that is not a per-minute rate limit (a daily or project quota, or a billing
    refusal): stop the row and report it as partial; never retry (Uzair's memo of 2026-11-08, section 1.3)."""


_first_429_logged = threading.Event()


def error_json(exc: BaseException) -> Any:
    """The response JSON a google-genai error carries: errors.APIError's `details`, or the Interactions client's
    compat error's `body` (parsed when it is text); None when there is none."""
    body = getattr(exc, "details", None)
    if body is None:
        body = getattr(exc, "body", None)
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            pass
    return body


def error_code(exc: BaseException) -> int | None:
    """The HTTP status of a google-genai error (`code` of errors.APIError, `status_code` of a compat error)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def error_body(exc: BaseException) -> dict:
    """What a google-genai error carries: code, status, message and the response JSON, for the log."""
    return {"type": type(exc).__name__, "code": error_code(exc), "status": getattr(exc, "status", None),
            "message": getattr(exc, "message", None) or str(exc), "details": error_json(exc)}


def is_permission_denied(exc: BaseException) -> bool:
    """An HTTP 403 (the policy refusal of 2026-11-08: "Request rejected in accordance with our terms of use and
    policies", permission_denied), whichever class carries it."""
    if error_code(exc) == 403:
        return True
    return "permission_denied" in str(exc).lower() or type(exc).__name__ == "PermissionDeniedError"


def _details_list(body: Any) -> list:
    """The `details` list of a Google error body ({"error": {"details": [...]}} or {"details": [...]})."""
    if not isinstance(body, dict):
        return []
    inner = body.get("error") if isinstance(body.get("error"), dict) else None
    details = inner.get("details") if inner is not None else None
    if details is None:
        details = body.get("details")
    return details if isinstance(details, list) else []


def quota_ids(body: Any) -> list[str]:
    """The quotaId of every google.rpc.QuotaFailure violation in the body (empty when there is none)."""
    out = []
    for entry in _details_list(body):
        if isinstance(entry, dict) and str(entry.get("@type", "")).endswith("QuotaFailure"):
            for v in entry.get("violations") or []:
                if isinstance(v, dict) and v.get("quotaId"):
                    out.append(str(v["quotaId"]))
    return out


def _retry_delay_s(body: Any) -> float | None:
    """The retryDelay (seconds) of a google.rpc.RetryInfo entry in the error's details, or None."""
    for entry in _details_list(body):
        if isinstance(entry, dict) and str(entry.get("@type", "")).endswith("RetryInfo"):
            delay = entry.get("retryDelay")
            if isinstance(delay, (int, float)):
                return float(delay)
            if isinstance(delay, dict) and delay.get("seconds") is not None:
                return float(delay["seconds"])
            m = re.fullmatch(r"\s*([0-9.]+)\s*s\s*", str(delay or ""))
            if m:
                return float(m.group(1))
    return None


def _per_minute(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in PER_MINUTE_MARKERS)


def classify_429(exc: BaseException) -> str:
    """"rate_limit" or "quota" for an HTTP 429 (module docstring). With QuotaFailure quota ids: rate_limit only when
    every quota id names a per-minute limit (one per-day or per-project id in the same body means quota). Without quota
    ids: rate_limit when the message names a per-minute limit or a RetryInfo of at most RETRY_INFO_MAX_S is present;
    else quota (a billing refusal, a body without detail: the memo stops on anything that is not a per-minute
    limit)."""
    body = error_json(exc)
    ids = quota_ids(body)
    if ids:
        return "rate_limit" if all(_per_minute(i) for i in ids) else "quota"
    message = str(getattr(exc, "message", None) or exc)
    if _per_minute(message) or (isinstance(body, dict) and _per_minute(json.dumps(body, default=str))):
        return "rate_limit"
    delay = _retry_delay_s(body)
    if delay is not None and delay <= RETRY_INFO_MAX_S:
        return "rate_limit"
    return "quota"


def route_credential(route: str) -> str | None:
    """How the route would authenticate ("api_key" | "adc"), or None when it has no credential."""
    if route == "aistudio":
        return "api_key" if _credential("GEMINI_API_KEY") else None
    if route == "vertex":
        if _credential("GOOGLE_CLOUD_PROJECT"):
            return "adc"
        return "api_key" if _credential("VERTEX_API_KEY") else None
    raise ValueError(f"unknown Gemini route {route!r} (expected one of {ROUTES})")


def sdk_http_options() -> Any:
    """google.genai types.HttpOptions for every client built here: the SDK's own retries on SDK_RETRY_STATUS_CODES
    only (no 429; module docstring)."""
    from google.genai import types

    return types.HttpOptions(retry_options=types.HttpRetryOptions(http_status_codes=list(SDK_RETRY_STATUS_CODES)))


def make_sdk(route: str) -> tuple[Any, str]:
    """(google.genai Client, route label) for the route, built with explicit arguments only and sdk_http_options()."""
    if os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is set; unset it (the SDK prefers it over GEMINI_API_KEY)")
    auth = route_credential(route)
    if auth is None:
        raise RuntimeError(f"no credential for Gemini route {route!r}")
    from google import genai

    http_options = sdk_http_options()
    if route == "aistudio":
        client = genai.Client(enterprise=False, api_key=_credential("GEMINI_API_KEY"), http_options=http_options)
        if client.vertexai:
            raise RuntimeError("the aistudio client came up in enterprise mode; refusing to use it")
        return client, "aistudio"
    if auth == "adc":
        client = genai.Client(enterprise=True, project=_credential("GOOGLE_CLOUD_PROJECT"), location=VERTEX_LOCATION,
                              http_options=http_options)
    else:
        client = genai.Client(enterprise=True, api_key=_credential("VERTEX_API_KEY"), http_options=http_options)
    if not client.vertexai:
        raise RuntimeError("the vertex client did not come up in enterprise mode; refusing to use it")
    return client, f"vertex:{auth}"


def _strip_additional_properties(schema: Any) -> Any:
    """A copy of the JSON schema without `additionalProperties` (not part of the subset Gemini documents)."""
    if isinstance(schema, dict):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


class GeminiClient:
    """One Gemini model on one route. Same `complete` interface as the other LLM clients."""

    provider = "google"

    def __init__(
        self,
        model: str,
        route: str = "aistudio",
        thinking_level: str = "low",
        max_output_tokens: int = 1500,
        *,
        cache: DiskCache,
        cost: CostLogger,
        prices: dict,
        sdk_client: Any = None,
        route_label: str | None = None,
    ) -> None:
        if route not in ROUTES:
            raise ValueError(f"unknown Gemini route {route!r} (expected one of {ROUTES})")
        self.model = model
        self.route = route
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.cache = cache
        self.cost = cost
        self.prices = prices or {}
        if sdk_client is not None:
            self.sdk, self.route_label = sdk_client, route_label or route
        else:
            self.sdk, self.route_label = make_sdk(route)

    def request_kwargs(self, prompt: str, system: str | None = None, schema: dict | None = None) -> dict:
        """The exact keyword arguments of interactions.create for this client."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "generation_config": {"thinking_level": self.thinking_level, "max_output_tokens": self.max_output_tokens},
            "store": False,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "text", "mime_type": "application/json",
                                         "schema": _strip_additional_properties(copy.deepcopy(schema))}
        if system is not None:
            kwargs["system_instruction"] = system
        return kwargs

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str,
        attempt: int = 1,
        schema: dict | None = None,
        tags: dict | None = None,
        cached_prefix: str | None = None,
    ) -> LLMResponse:
        """Complete one user prompt (disk cache first). The route is part of the cache key."""
        prompt = (cached_prefix or "") + prompt
        payload = _payload(
            "gemini", self.model, prompt, system, self.max_output_tokens, None, None,
            {"thinking_level": self.thinking_level, "route": self.route}, "", attempt, schema=schema, tags=tags,
        )

        def call() -> LLMResponse:
            kwargs = self.request_kwargs(prompt, system, schema)

            def once() -> Any:
                try:
                    return self.sdk.interactions.create(**kwargs)
                except Exception as exc:
                    code = error_code(exc)
                    if code == 402:
                        raise BalanceExhausted(
                            f"gemini {self.model} on route {self.route_label}: HTTP 402, the balance is exhausted"
                        ) from exc
                    if code == 429:
                        kind = classify_429(exc)
                        body = json.dumps(error_body(exc), default=str, ensure_ascii=False)
                        if not _first_429_logged.is_set():
                            _first_429_logged.set()
                            logger.warning("gemini %s [%s]: first HTTP 429 of this process, classified %s; body: %s",
                                           self.model, self.route_label, kind, body[:4000])
                        if kind == "quota":
                            logger.error("gemini %s [%s]: HTTP 429 that is not a per-minute limit; stopping without "
                                         "a retry: %s", self.model, self.route_label, body[:4000])
                            raise QuotaExhausted(
                                f"gemini {self.model} on route {self.route_label}: HTTP 429 "
                                f"{getattr(exc, 'status', None) or ''} that is not a per-minute rate limit: "
                                f"{getattr(exc, 'message', None) or exc}"
                            ) from exc
                    raise

            r = _with_backoff(once, f"gemini {self.model} [{self.route_label}]")
            text = r.output_text or ""
            status = str(r.status)
            if status != "completed":
                logger.warning("gemini %s [%s]: interaction status %s (purpose=%s)", self.model, self.route_label,
                               status, purpose)
            usage_raw: dict = {}
            if r.usage is not None:
                dumped = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
                usage_raw = {k: v for k, v in dumped.items() if v is not None}
            in_tok = int(usage_raw.get("total_input_tokens") or 0)
            out_tok = int(usage_raw.get("total_output_tokens") or 0)
            thought = int(usage_raw.get("total_thought_tokens") or 0)
            cached_tok = int(usage_raw.get("total_cached_tokens") or 0)
            p = self.prices.get(self.model)
            parts = None
            if p is not None:
                # Thought tokens are billed as output. Cached input (implicit caching) at the table's
                # implicit_cache_read price when it has one, else at the input rate; the cached count is taken as a
                # subset of the input count (module docstring), with the upper bound when a response says otherwise.
                cache_price = p.get("implicit_cache_read", p["input"])
                if cached_tok > in_tok:
                    logger.warning("gemini %s [%s]: total_cached_tokens %d exceeds total_input_tokens %d; pricing "
                                   "all input at the full rate and the cached tokens on top (upper bound)",
                                   self.model, self.route_label, cached_tok, in_tok)
                    uncached = in_tok
                else:
                    uncached = in_tok - cached_tok
                parts = {"input": uncached * p["input"] / 1e6, "cache_read": cached_tok * cache_price / 1e6,
                         "cache_write": 0.0, "output": (out_tok + thought) * p["output"] / 1e6}
            else:
                logger.warning("no price for model %r: cost_usd will be null", self.model)
            return LLMResponse(
                text=text,
                model=str(getattr(r, "model", None) or self.model),
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=None if parts is None else sum(parts.values()),
                cached=False,
                stop_reason=status,
                request=payload,
                cache_read_input_tokens=cached_tok,
                thought_tokens=thought,
                cost_parts=parts,
                route=self.route_label,
                usage_raw=usage_raw,
            )

        return _complete_cached(
            cache=self.cache,
            cost=self.cost,
            provider="google",
            model=self.model,
            payload=payload,
            purpose=purpose,
            est_input_tokens=estimate_tokens((system or "") + prompt),
            max_tokens=self.max_output_tokens,
            call=call,
            route=self.route,
        )


def ping(client: GeminiClient, purpose: str = PROBE_PURPOSE) -> LLMResponse:
    """One tiny live call on the client's route (memo section 1.2's probe): PROBE_PROMPT with a nonce tag, so the
    disk cache never serves it and a second probe is live too; logged in cost.jsonl under `purpose` like any call
    (about 0.00005 USD). Raises whatever the API raises (a 403 is the caller's stop signal)."""
    nonce = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return client.complete(PROBE_PROMPT, purpose=purpose, tags={"probe": nonce})


def list_model_names(route: str) -> list[str]:
    """The model names the route lists (models.list of the SDK client; no cost), sorted; "models/" prefixes kept as
    the API returns them. Used once to verify that the aistudio route names the Part 2 models (memo section 1.1)."""
    sdk, _ = make_sdk(route)
    return sorted(str(getattr(m, "name", m)) for m in sdk.models.list())


def make_gemini_client(spec: dict, cache: DiskCache, cost: CostLogger, prices: dict) -> GeminiClient | None:
    """Client for a `provider: gemini` spec, or None when the spec's route has no credential."""
    route = spec.get("route", "aistudio")
    if route_credential(route) is None:
        return None
    return GeminiClient(
        spec["model"],
        route=route,
        thinking_level=spec.get("thinking_level", "low"),
        max_output_tokens=spec.get("max_output_tokens", 1500),
        cache=cache,
        cost=cost,
        prices=prices,
    )
