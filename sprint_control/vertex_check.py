"""The one check of the vertex route (sprint_control/README.md, section 3): one request of a few tokens to
Gemini 3.8 Flash through application default credentials. Made once; the result decides whether the two Gemini
judges take part.

What is sent: the main run's request for this judge (overclaim.graders.gemini.GeminiClient.request_kwargs: the model
name, thinking level low, at most 1,500 output tokens, nothing stored) with the prompt "Reply with the single word
OK." in the place of the scoring prompt.

One request means one. The main run's client would retry (judge._with_backoff on 429 and 5xx, and the SDK's
Interactions client on 408, 409, 429 and 5xx underneath it), so this script does not use it. It calls the SDK once:

- the SDK's own retries are switched off on the built client and read back before the request;
- the SDK's automatic continuation, which would send a second request after an unfinished reply, is switched off
  for this call;
- the call has a time limit, and the script counts the HTTP requests that leave;
- before the request leaves, the script creates results/sprint_control/vertex_check.json (exclusive create) saying
  that the check is being sent, so that a crash or an interrupt cannot lead to a second request later. The result
  overwrites it.

"Answers" means that the request came back with HTTP 200 and a reply. Any error is the answer "no": it is recorded
and nothing is tried again, on this or another route. One case is not an answer of either kind: the SDK failed
before any HTTP request left (a local fault). The file then records zero requests, and only then may the script be
started again.

Before the request, with no model request: GOOGLE_API_KEY must be unset, and so must the variables that would send
the request to another host or print its headers; GOOGLE_CLOUD_PROJECT must be poverclaim and GOOGLE_CLOUD_LOCATION
global; the route must resolve to application default credentials, not to a key; and those credentials must be able
to mint a token (a request to Google's token endpoint, not to a model). If one of these fails the script stops
before the request and writes nothing, so the one check is not spent on a local fault.

The script refuses to run while a tracked file that shapes a call differs from the commit: the declaration is
committed before the first call. No key, token or credential is printed or written.

Run: uv run python sprint_control/vertex_check.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import control_common as K
from run_control import committed_or_exit, forbidden_environment, judge_spec, limits, load_config, only_copy_or_exit

TIME_LIMIT_S = 120.0


def preconditions(check: dict) -> str:
    """The project id, after every local check passed. SystemExit (nothing written) when one fails."""
    from overclaim.graders import gemini
    from overclaim.paths import load_env

    load_env()
    bad = forbidden_environment()
    if bad:
        raise SystemExit(f"these variables must be unset: {', '.join(bad)}. The check was not sent.")
    project, location = os.environ.get("GOOGLE_CLOUD_PROJECT", ""), os.environ.get("GOOGLE_CLOUD_LOCATION", "")
    if project != check["project"] or location != check["location"] or gemini.VERTEX_LOCATION != check["location"]:
        raise SystemExit(f"GOOGLE_CLOUD_PROJECT must be {check['project']} and GOOGLE_CLOUD_LOCATION "
                         f"{check['location']}. The check was not sent.")
    if gemini.route_credential("vertex") != "adc":
        raise SystemExit("the vertex route would not use application default credentials. The check was not sent.")
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
    except Exception as exc:
        raise SystemExit("the application default credentials cannot mint a token "
                         f"({type(exc).__name__}). The check was not sent; run `gcloud auth "
                         "application-default login` and start the check again.") from None
    if not credentials.valid:
        raise SystemExit("the application default credentials are not valid. The check was not sent.")
    return project


def one_request_client(project: str):
    """(SDK client for the vertex route as gemini.make_sdk builds it, the list its HTTP requests are counted in),
    with every retry of the SDK switched off. AssertionError if a retry could still happen."""
    from google import genai
    from google.genai import types

    from overclaim.graders import gemini

    client = genai.Client(enterprise=True, project=project, location=gemini.VERTEX_LOCATION,
                          http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    if not client.vertexai:
        raise AssertionError("the client did not come up in enterprise mode")
    interactions = client.interactions
    # google-genai 2.34 turns `attempts` into the Interactions client's number of retries, so "one attempt" would be
    # one retry. The built configuration is set to no retry and read back.
    retry = interactions.sdk_configuration.retry_config
    retry.max_retries = 0
    if interactions.sdk_configuration.retry_config.max_retries != 0:
        raise AssertionError("the Interactions client would still retry")
    sent: list[str] = []
    http = client._api_client._httpx_client
    if interactions.sdk_configuration.client is not http:
        raise AssertionError("the Interactions client does not send through the counted HTTP client")
    http.event_hooks["request"].append(lambda request: sent.append(f"{request.method} {request.url.host}"
                                                                   f"{request.url.path}"))
    return client, sent


def claim_the_check(marker: dict) -> None:
    """Create the result file before the request leaves (exclusive create), or SystemExit when the check was already
    made. A file that records zero HTTP requests (a local fault before anything left) is the one case that may be
    replaced."""
    K.OUT.mkdir(parents=True, exist_ok=True)
    earlier = K.read_json(K.VERTEX_CHECK_FILE)
    if earlier is not None:
        if earlier.get("http_requests_counted") == 0 and earlier.get("answered") is None:
            K.VERTEX_CHECK_FILE.unlink()                 # nothing left the machine: the check was not made
        else:
            raise SystemExit(f"{K.VERTEX_CHECK_FILE.relative_to(K.REPO)} exists: the check is made once")
    with open(K.VERTEX_CHECK_FILE, "x", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    from overclaim.graders import gemini, judge

    cfg = load_config()
    check = cfg["vertex_check"]
    lock = only_copy_or_exit()                       # no second copy of this check and no paid stage beside it
    earlier = K.read_json(K.VERTEX_CHECK_FILE)
    if earlier is not None and not (earlier.get("http_requests_counted") == 0 and earlier.get("answered") is None):
        raise SystemExit(f"{K.VERTEX_CHECK_FILE.relative_to(K.REPO)} exists: the check is made once")
    commit = committed_or_exit()
    spec = judge_spec(cfg, {"judge": check["judge"], "route": check["route"]})
    if spec["provider"] != "gemini" or spec["route"] != "vertex":
        raise SystemExit("the check is for a Gemini judge on the vertex route")
    project = preconditions(check)
    client, sent = one_request_client(project)
    shape = gemini.GeminiClient(spec["model"], route="vertex", thinking_level=spec.get("thinking_level", "low"),
                                max_output_tokens=spec.get("max_output_tokens", 1500), cache=None, cost=None,
                                prices=cfg["prices"], sdk_client=client, route_label="vertex:adc")
    kwargs = shape.request_kwargs(check["prompt"])
    result: dict = {"at": datetime.now(timezone.utc).isoformat(), "commit": commit, "route": "vertex:adc",
                    "project": project, "location": gemini.VERTEX_LOCATION, "model": spec["model"],
                    "request": dict(kwargs), "sdk_retries": 0, "automatic_continuation": False,
                    "time_limit_s": TIME_LIMIT_S, "answered": None, "http_requests_counted": None,
                    "outcome": "being sent; no result recorded yet"}
    claim_the_check(result)
    usage: dict = {}
    try:
        try:
            reply = client.interactions.create(**kwargs, automatic_continuation=False,
                                               timeout=TIME_LIMIT_S)        # the one request
        except Exception as exc:
            body = gemini.error_body(exc)
            if not sent:                               # the SDK failed before any HTTP request left: not an answer
                result.update(answered=None, outcome=f"not sent: {body['type']} before any request left")
            else:
                result.update(answered=False, outcome=f"error {body.get('code')} {body.get('status') or ''}".strip(),
                              error={"type": body["type"], "code": body["code"], "status": body["status"],
                                     "message": str(body["message"])[:1000], "details": body["details"]})
        else:
            # The route answered: recorded first, before any detail of the reply is read.
            result.update(answered=True, outcome="HTTP 200")
            try:
                if reply.usage is not None:
                    dumped = reply.usage.model_dump() if hasattr(reply.usage, "model_dump") else dict(reply.usage)
                    usage = {k: v for k, v in dumped.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)}
                result.update(outcome=f"HTTP 200, status {reply.status}", reply_text=(reply.output_text or "")[:200],
                              reply_model=str(getattr(reply, "model", "") or ""), usage=usage)
            except Exception as exc:                   # the answer stands; only its details could not be read
                result["reply_details_unread"] = type(exc).__name__
    finally:
        # Whatever happened, including an interrupt: the file says how many requests left.
        result["http_requests_counted"] = len(sent)
        result["http_requests"] = list(sent)
        if result["answered"] is None and sent:
            result.update(answered=False, outcome="interrupted after the request left; no reply recorded")
        in_tok, out_tok = int(usage.get("total_input_tokens") or 0), int(usage.get("total_output_tokens") or 0)
        thought, cached = int(usage.get("total_thought_tokens") or 0), int(usage.get("total_cached_tokens") or 0)
        price = cfg["prices"][spec["model"]]
        cost_usd = ((in_tok - min(cached, in_tok)) * price["input"] + min(cached, in_tok)
                    * price.get("implicit_cache_read", price["input"]) + (out_tok + thought) * price["output"]) / 1e6
        result["cost_usd"] = cost_usd
        result = json.loads(json.dumps(result, default=str))       # an error body may hold objects; keep it writable
        K.write_json(K.VERTEX_CHECK_FILE, result)
        if sent:                                       # a request left: it counts against the Google cap
            judge.CostLogger(str(K.OUT), prices=cfg["prices"], caps=limits(cfg)).record(
                provider="google", model=spec["model"], purpose="vertex_check", input_tokens=in_tok,
                output_tokens=out_tok, cost_usd=cost_usd,
                extra={"cache_read_input_tokens": cached, "cache_creation_input_tokens": 0,
                       "thought_tokens": thought, "cost_parts": None, "route": "vertex:adc",
                       "upstream_provider": None})
    print(f"vertex check: answered = {result['answered']}; {result['outcome']}; "
          f"{result['http_requests_counted']} HTTP request(s) left; cost {cost_usd:.6f} USD")
    if result["answered"] is False:
        print("the two Gemini judges do not take part; nothing is tried again and no other route is used")
    for key in ("reply_text", "reply_model", "usage", "error"):
        if key in result:
            print(f"{key}: {json.dumps(result[key], ensure_ascii=False, default=str)[:1500]}")
    lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
