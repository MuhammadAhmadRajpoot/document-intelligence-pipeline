import httpx
import pytest
from openai import RateLimitError

from docintel.confidence import score_fields
from docintel.llm.client import LLMError, parse_json_object, rate_limit_wait_s
from docintel.models import Page, TextLine, TextSource
from docintel.schemas import Receipt


def rate_limit_error(message: str, headers: dict[str, str] | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers=headers or {})
    return RateLimitError(message, response=response, body=None)


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        parse_json_object("no json here")
    with pytest.raises(LLMError):
        parse_json_object("{broken: }")


def test_rate_limit_wait_prefers_server_hints():
    assert rate_limit_wait_s(rate_limit_error("x", {"retry-after": "3"}), 0) == 3.5
    assert rate_limit_wait_s(rate_limit_error("Please try again in 250ms."), 0) == 0.75
    assert rate_limit_wait_s(rate_limit_error("Please try again in 9.9975s."), 0) == 10.4975
    assert rate_limit_wait_s(rate_limit_error("Please try again in 6m43.5s."), 0) == 404.0
    assert rate_limit_wait_s(rate_limit_error("slow down"), 3) == 8.0


def test_short_currency_aliases_ground_as_whole_words():
    def page(text: str) -> Page:
        line = TextLine(text=text, bbox=(0, 0, 10, 10), confidence=1.0)
        return Page(index=0, width=10, height=10, source=TextSource.OCR, lines=[line])

    receipt = Receipt.model_validate({"currency": "PKR", "total": 10})
    assert score_fields(receipt, [page("TOTAL Rs 4,500.00")], [])["currency"].grounded
    assert not score_fields(receipt, [page("Partners for 3 hours")], [])["currency"].grounded
