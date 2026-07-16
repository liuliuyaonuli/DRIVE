"""Core DRIVE data model and scenario-aware dual-level retrieval.

The paper represents every skill as ``(k, d_k)`` where
``d_k = <U_k, W_k>``.  ``U_k`` describes the applicable page context and
``W_k`` describes the semantic task scenario.  This module is deliberately
model-independent: structural filtering and compact descriptor ranking happen
locally, so retrieval does not inject the full library into an LLM prompt.

Legacy skill artifacts are normalized at the boundary.  This keeps old public
artifacts runnable while all newly generated artifacts use the exact DRIVE
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse
import copy
import fnmatch
import math
import re


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}", re.IGNORECASE)
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "before", "but",
    "can", "current", "does", "for", "from", "have", "into", "its",
    "more", "not", "of", "on", "only", "or", "page", "please", "should",
    "that", "the", "their", "then", "this", "to", "use", "using", "was",
    "were", "what", "when", "where", "which", "with", "you", "your",
}
_ACTION_GROUPS = {
    "create": {"create", "publish", "submit", "upload"},
    "edit": {"change", "edit", "modify", "rename", "set", "update"},
    "delete": {"cancel", "delete", "remove"},
    "vote": {"downvote", "like", "upvote", "vote"},
    "reply": {"comment", "reply", "respond"},
    "retrieve": {
        "count", "extract", "extracts", "find", "get", "identify", "identifies",
        "information", "list", "retrieval", "search", "show", "tell",
    },
    "subscribe": {"follow", "subscribe", "unsubscribe"},
    "route": {"directions", "distance", "drive", "route", "travel"},
    "cart": {"buy", "cart", "checkout", "purchase"},
}
_OBJECT_GROUPS = {
    "forum": {"forum", "forums", "subreddit"},
    "post": {"post", "posts", "submission", "thread"},
    "comment": {"comment", "comments", "reply", "replies"},
    "user": {"account", "bio", "biography", "profile", "user", "username"},
    "product": {"item", "product", "products", "sku"},
    "order": {"order", "orders"},
    "issue": {"issue", "issues", "ticket"},
    "project": {"project", "repository", "repo"},
    "place": {"location", "map", "place", "poi", "route"},
    "book": {"book", "books", "recommendation"},
}
_GENERIC_PROPER_PHRASES = {"Press Enter"}
_PROPER_NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b")
_INITIALLED_NAME_RE = re.compile(r"\b(?:[A-Z]\.\s*){1,4}[A-Z][a-z]+\b")
_CONTEXTUAL_SINGLE_ENTITY_RE = re.compile(
    r"\b(?P<prefix>(?i:from|to|at|near|around|for|by|of|in))\s+(?P<entity>[A-Z][a-z]+)\b",
)
_REASONING_REUSE_PROFILE = {
    "S": "semantically_related_tasks_execution_stages_and_decision_conditions",
    "E": ["semantic_compatibility"],
    "I": "structured_context_augmentation",
    "F": ["subsequent_decision_trace", "terminal_task_outcome"],
}


def _action_groups(tokens: set[str]) -> set[str]:
    return {
        group
        for group, members in _ACTION_GROUPS.items()
        if tokens & members
    }


def _object_groups(tokens: set[str]) -> set[str]:
    return {
        group
        for group, members in _OBJECT_GROUPS.items()
        if tokens & members
    }


def _as_text(value: Any) -> str:
    """Return a compact textual view of an observation or descriptor value."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if "text" in value:
            return _as_text(value["text"])
        return " ".join(_as_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_text(item) for item in value)
    return str(value)


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        return [_as_text(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def tokenize(value: Any) -> set[str]:
    """Tokenize compact task/page text for descriptor-level matching."""

    tokens = set()
    for token in _TOKEN_RE.findall(_as_text(value).lower().replace("_", " ")):
        normalized = token.strip("-_")
        if len(normalized) > 1 and normalized not in _STOP_WORDS:
            tokens.add(normalized)
    return tokens


def canonical_site(site: Optional[str]) -> str:
    """Normalize historical library names to their WebArena site name."""

    if not site:
        return ""
    value = str(site).lower().strip().replace("-", "_")
    value = value.split("/")[0]
    if "shopping_admin" in value or value in {"admin", "cms"}:
        return "shopping_admin"
    if "reddit" in value or "postmill" in value or "forum" == value:
        return "reddit"
    if "gitlab" in value or value == "git":
        return "gitlab"
    if "shopping" in value or value == "shop":
        return "shopping"
    if "map" in value or "openstreetmap" in value:
        return "map"
    if "miniwob" in value:
        return "miniwob"
    # Strip common experiment suffixes without damaging unknown site names.
    return re.sub(r"_(?:\d+%?|train|test|mixed).*$", "", value)


def infer_site_from_url(url: str) -> str:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    combined = f"{host} {path}"
    if "admin" in combined and ("shop" in combined or "magento" in combined):
        return "shopping_admin"
    if any(value in combined for value in ("reddit", "postmill")):
        return "reddit"
    if "gitlab" in combined:
        return "gitlab"
    if any(value in combined for value in ("openstreetmap", "map.local", "maps.")):
        return "map"
    if any(value in combined for value in ("shopping", "shop.local", "magento")):
        return "shopping"
    if "miniwob" in combined:
        return "miniwob"
    return ""


def _page_type(url: str, observation: str) -> str:
    path = urlparse(url or "").path.lower()
    text = observation.lower()[:5000]
    rules = (
        ("checkout", ("/checkout", "shopping cart")),
        ("product", ("/product", "add to cart")),
        ("forum", ("/f/", "forum")),
        ("post", ("/post/", "/comments/", "comment")),
        ("profile", ("/user/", "/profile", "profile")),
        ("settings", ("/settings", "settings")),
        ("issue", ("/issues", "issue")),
        ("merge_request", ("/merge_requests", "merge request")),
        ("route", ("/directions", "route")),
        ("search", ("/search", "search")),
    )
    for page_type, cues in rules:
        if any(cue in path or cue in text for cue in cues):
            return page_type
    return "generic"


@dataclass(frozen=True)
class PageContext:
    """Compact local page context ``C_t = ExtractContext(o_t)``."""

    url: str
    observation: str
    site: str
    page_type: str
    ui_tokens: frozenset[str]

    def as_feedback_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "site": self.site,
            "page_type": self.page_type,
            "observation_snippet": self.observation[:2000],
        }


def extract_context(
    url: str,
    observation: Any,
    *,
    site_hint: Optional[str] = None,
) -> PageContext:
    text = _as_text(observation)
    site = infer_site_from_url(url) or canonical_site(site_hint)
    return PageContext(
        url=url or "",
        observation=text,
        site=site,
        page_type=_page_type(url or "", text),
        ui_tokens=frozenset(tokenize(text[:12000])),
    )


def _descriptor_site_values(value: Any) -> set[str]:
    return {canonical_site(item) for item in _listify(value) if canonical_site(item)}


def _url_pattern_matches(url: str, pattern: str) -> bool:
    """Match exact, glob, regex-like-template, or path URL descriptors."""

    if not pattern:
        return False
    pattern = pattern.strip()
    if pattern in {"*", "/*", "any"}:
        return True

    parsed_url = urlparse(url or "")
    parsed_pattern = urlparse(pattern)
    target = url or ""
    if parsed_pattern.scheme or parsed_pattern.netloc:
        if parsed_pattern.netloc and parsed_pattern.netloc.lower() != parsed_url.netloc.lower():
            return False
        target = parsed_url.path or "/"
        pattern = parsed_pattern.path or "/"
        if parsed_pattern.query:
            target += "?" + parsed_url.query
            pattern += "?" + parsed_pattern.query
    elif pattern.startswith("/"):
        target = parsed_url.path or "/"

    # ``{name}`` is the descriptor convention for a variable path component.
    escaped = re.escape(pattern)
    templated = re.sub(r"\\\{[^{}]+\\\}", r"[^/?#]+", escaped)
    templated = templated.replace(r"\*", ".*")
    if re.fullmatch(templated, target, flags=re.IGNORECASE):
        return True
    return fnmatch.fnmatch(target.lower(), pattern.lower()) or pattern.lower() in target.lower()


def structural_match(context: PageContext, applicable_context: Mapping[str, Any]) -> bool:
    """Evaluate ``Match(C_t, U_k)`` without looking at task semantics.

    Site, explicitly required URL patterns, page types, and UI cues are hard
    constraints.  Historical ``url_patterns`` are treated as soft structural
    hints unless the artifact declares ``url_match: required``; old generators
    often stored a trajectory's concrete URL even for skills that begin by
    navigating to their own stable start page.
    """

    u_k = applicable_context or {}
    sites = _descriptor_site_values(u_k.get("sites", u_k.get("site")))
    if sites and context.site and canonical_site(context.site) not in sites:
        return False

    required_patterns = _listify(u_k.get("required_url_patterns"))
    url_patterns = _listify(u_k.get("url_patterns"))
    url_policy = str(u_k.get("url_match", u_k.get("url_policy", "compatible_site"))).lower()
    if required_patterns and not any(_url_pattern_matches(context.url, item) for item in required_patterns):
        return False
    if url_patterns and url_policy in {"required", "strict", "matching_page"}:
        if not any(_url_pattern_matches(context.url, item) for item in url_patterns):
            return False

    page_types = {item.lower() for item in _listify(u_k.get("page_types", u_k.get("page_type")))}
    if page_types and context.page_type.lower() not in page_types:
        return False

    required_cues = _listify(
        u_k.get("required_ui_cues", u_k.get("required_elements", u_k.get("ui_cues")))
    )
    if required_cues:
        observation = context.observation.lower()
        if not any(cue.lower() in observation for cue in required_cues):
            return False
    return True


def normalize_descriptor(
    raw_descriptor: Optional[Mapping[str, Any]],
    *,
    site: str = "",
    url_patterns: Any = None,
    semantic_keywords: Any = None,
    task_intent: str = "",
    task_family: str = "",
    scenario_description: str = "",
) -> dict[str, dict[str, Any]]:
    """Normalize a descriptor to the exact ``{U_k, W_k}`` schema."""

    raw = copy.deepcopy(dict(raw_descriptor or {}))
    u_k = dict(raw.get("U_k") or raw.get("U") or {})
    w_k = dict(raw.get("W_k") or raw.get("W") or {})

    if site and not u_k.get("site") and not u_k.get("sites"):
        u_k["site"] = canonical_site(site)
    if not u_k.get("url_patterns") and url_patterns:
        u_k["url_patterns"] = _listify(url_patterns)
    u_k.setdefault("url_patterns", [])
    # Legacy concrete trajectory URLs are hints, not hard applicability gates.
    u_k.setdefault("url_match", "compatible_site")

    if task_intent and not w_k.get("task_intent"):
        w_k["task_intent"] = task_intent
    if task_family and not w_k.get("task_family"):
        w_k["task_family"] = task_family
    if scenario_description and not w_k.get("scenario_description"):
        w_k["scenario_description"] = scenario_description
    if not w_k.get("semantic_keywords") and semantic_keywords:
        w_k["semantic_keywords"] = _listify(semantic_keywords)
    w_k.setdefault("semantic_keywords", [])
    return {"U_k": u_k, "W_k": w_k}


def interaction_descriptor(skill: Any, *, site_hint: str = "") -> dict[str, dict[str, Any]]:
    metadata = getattr(skill, "metadata", {}) or {}
    name = getattr(skill, "name", "")
    doc = getattr(skill, "doc", "")
    descriptor = normalize_descriptor(
        metadata.get("scenario_descriptor"),
        site=metadata.get("site", site_hint),
        url_patterns=metadata.get("url_patterns"),
        semantic_keywords=metadata.get("scenario_keywords") or tokenize(f"{name} {doc}"),
        task_intent=metadata.get("task_intent", ""),
        task_family=metadata.get("task_family", ""),
        scenario_description=metadata.get("scenario_description", "") or str(doc).split("\n")[0],
    )
    # Ensure a legacy skill still has a semantic retrieval key.
    if not descriptor["W_k"].get("task_intent"):
        descriptor["W_k"]["task_intent"] = str(doc).split("\n")[0] or name.replace("_", " ")
    if not descriptor["W_k"].get("semantic_keywords"):
        descriptor["W_k"]["semantic_keywords"] = sorted(tokenize(f"{name} {doc}"))[:20]
    return descriptor


def _join_guidance(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return " ".join(item.strip() for item in _listify(value) if item.strip())


def generalize_reasoning_text(value: Any) -> str:
    """Remove evaluator labels and instance-answer leakage from reusable rules."""

    text = _join_guidance(value)
    if not text:
        return ""
    replacements = (
        (r"\b(?:ground[- ]?truth|reference answers?|expected answers?|correct answers?)\b", "task requirements"),
        (r"\bmust_include\b", "task-requested"),
        (r"\bstring_match\b", "exact-format check"),
        (r"\bprogram_html\b", "page-state check"),
        (r"\burl_match\b", "URL-state check"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "<page_context>", text)
    # Treat web paths as episodic context, but preserve general formats such
    # as ``MM/DD/YYYY``: their slash segments begin with uppercase placeholders
    # rather than a normal URL path component.
    text = re.sub(r"(?:/[a-z][A-Za-z0-9_.{}-]*){2,}", "<page_context>", text)
    text = re.sub(
        r"used the phrase\s*[\"'][^\"']{1,80}[\"']\s*instead of (?:the )?required\s*[\"'][^\"']{1,80}[\"']\s*format",
        "used an output form that did not match the required format",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"task requirements\s+(?:was|were|is|are|expects?|expected)\s*[\"'][^\"']{1,100}[\"']",
        "the required output semantics were not satisfied",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\(\s*(?:e\.g\.|for example),?\s*[\"'][^)]{1,160}\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:required|expected)\s+(?:books?|items?|names?|values?)\s*\([^)]{1,200}\)",
        "task-requested items",
        text,
        flags=re.IGNORECASE,
    )
    # M-B-V is reusable guidance, not an episodic record. Quoted values and
    # concrete numeric answers are task-instance entities and must not survive
    # admission into K^r.
    text = re.sub(r"([\"'])[^\"']{1,160}\1", "<task_value>", text)
    text = re.sub(
        r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:h(?:ours?|rs?)?|m(?:in(?:ute)?s?)?|hours?|miles?|km|items?|posts?|comments?)?\b",
        "<task_value>",
        text,
        flags=re.IGNORECASE,
    )

    # Proper *multi-word* names in a lesson are usually the failed task's
    # answer or input (places, users, products).  Do not treat a single
    # capitalized sentence-start verb such as "Review" as an entity.
    def replace_proper_name(match: re.Match[str]) -> str:
        phrase = match.group(0)
        return phrase if phrase in _GENERIC_PROPER_PHRASES else "<task_entity>"

    text = _PROPER_NAME_RE.sub(replace_proper_name, text)
    text = _INITIALLED_NAME_RE.sub("<task_entity>", text)

    def replace_contextual_entity(match: re.Match[str]) -> str:
        entity = match.group("entity")
        if entity in {"A", "An", "The", "This", "That"}:
            return match.group(0)
        return f"{match.group('prefix')} <task_entity>"

    text = _CONTEXTUAL_SINGLE_ENTITY_RE.sub(replace_contextual_entity, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_reasoning_skill(lesson: Mapping[str, Any], *, site_hint: str = "") -> dict[str, Any]:
    """Return a reasoning entry with exact ``<M, B, V>`` and descriptor fields."""

    normalized = copy.deepcopy(dict(lesson))
    legacy = dict(normalized.get("task_lessons") or normalized.get("lessons") or {})
    reasoning = dict(normalized.get("reasoning_skill") or {})
    mistake = _join_guidance(
        reasoning.get("M")
        or legacy.get("why_failed")
        or legacy.get("decision_mistakes")
        or legacy.get("failure_summary")
        or normalized.get("lesson")
    )
    behavior = _join_guidance(
        reasoning.get("B")
        or legacy.get("better_strategy")
        or legacy.get("task_level_tips")
        or legacy.get("avoid_tips")
    )
    verification = _join_guidance(
        reasoning.get("V")
        or legacy.get("verification_strategy")
        or legacy.get("expected_final_state")
        or legacy.get("failure_signals")
    )
    normalized["reasoning_skill"] = {
        "M": generalize_reasoning_text(mistake)
        or "A related prior decision pattern led to task failure.",
        "B": generalize_reasoning_text(behavior)
        or "Re-evaluate the task constraints and choose a corrected strategy.",
        "V": generalize_reasoning_text(verification)
        or "Verify the final page state and every task requirement before stopping.",
    }

    normalized["scenario_descriptor"] = normalize_descriptor(
        normalized.get("scenario_descriptor"),
        site=normalized.get("site", site_hint),
        url_patterns=legacy.get("url_patterns"),
        semantic_keywords=legacy.get("scenario_keywords"),
        task_intent=legacy.get("task_pattern", ""),
        task_family=normalized.get("task_family", legacy.get("task_family", "")),
        scenario_description=legacy.get("scenario_description", ""),
    )
    # Reasoning applicability is semantic. Page URLs/selectors belong to K^i.
    u_k = normalized["scenario_descriptor"]["U_k"]
    for field in (
        "url_patterns",
        "required_url_patterns",
        "required_ui_cues",
        "required_elements",
        "ui_cues",
        "page_context",
    ):
        u_k.pop(field, None)
    u_k["url_patterns"] = []
    u_k["url_match"] = "compatible_site"
    w_k = normalized["scenario_descriptor"]["W_k"]
    # ``K^r`` must not retain an episodic task instruction.  The task family
    # and generic semantic keywords are enough for scenario-level retrieval;
    # concrete intent text belongs to the current query q, not the library.
    task_family = _join_guidance(
        w_k.get("task_family") or normalized.get("task_family") or legacy.get("task_family")
    )
    if task_family:
        w_k["task_family"] = task_family
        w_k["task_intent"] = task_family.replace("_", " ") + " task"
    else:
        w_k.pop("task_intent", None)
    w_k.pop("scenario_description", None)
    if w_k.get("semantic_keywords"):
        generalized_keywords = []
        for keyword in _listify(w_k["semantic_keywords"]):
            value = generalize_reasoning_text(keyword)
            if (
                value
                and "<task_value>" not in value
                and "<task_entity>" not in value
                and value not in generalized_keywords
            ):
                generalized_keywords.append(value)
        w_k["semantic_keywords"] = generalized_keywords[:20]

    # Keep only non-episodic provenance around the exact K^r fields.  In
    # particular, do not serialize raw task texts, example answers, or legacy
    # lesson payloads beside the generalized M/B/V representation.
    retained_fields = {
        key: copy.deepcopy(normalized[key])
        for key in (
            "skill_id", "cluster_id", "skill_name", "task_family", "cluster_size",
            "skills", "source_sites", "merged_from", "statistics",
        )
        if key in normalized
    }
    normalized = retained_fields
    normalized.setdefault(
        "skill_id",
        str(normalized.get("cluster_id", normalized.get("skill_name", "reasoning_skill"))),
    )
    normalized["reasoning_skill"] = {
        "M": generalize_reasoning_text(mistake)
        or "A related prior decision pattern led to task failure.",
        "B": generalize_reasoning_text(behavior)
        or "Re-evaluate the task constraints and choose a corrected strategy.",
        "V": generalize_reasoning_text(verification)
        or "Verify the final page state and every task requirement before stopping.",
    }
    normalized["scenario_descriptor"] = {
        "U_k": u_k,
        "W_k": w_k,
    }
    # Table 1's shared reuse profile P=<S,E,I,F> makes the different
    # transfer contracts explicit without mixing their actual skill content.
    normalized["reuse_profile"] = copy.deepcopy(_REASONING_REUSE_PROFILE)
    stats = dict(normalized.get("statistics") or {})
    usage = int(stats.get("N", stats.get("usage_count", 0)) or 0)
    success = int(stats.get("S", stats.get("success_count", 0)) or 0)
    smoothing = float(stats.get("lambda", 1.0) or 1.0)
    stats.update(
        {
            "N": usage,
            "S": min(success, usage),
            "lambda": smoothing,
            "rho": (min(success, usage) + smoothing) / (usage + 2 * smoothing),
        }
    )
    normalized["statistics"] = stats
    return normalized


def merge_reasoning_group(
    lessons: Sequence[Mapping[str, Any]], *, site_hint: str = ""
) -> dict[str, Any]:
    """Merge a similar reasoning group into one generalized ``<M,B,V>`` rule."""

    normalized = [normalize_reasoning_skill(item, site_hint=site_hint) for item in lessons]
    if not normalized:
        raise ValueError("Cannot merge an empty reasoning-skill group")

    def utility(item: Mapping[str, Any]) -> tuple[float, int]:
        stats = item.get("statistics", {})
        return float(stats.get("rho", 0.5)), int(stats.get("N", 0))

    base = copy.deepcopy(max(normalized, key=utility))

    def sentence_candidates(field: str) -> list[str]:
        result = []
        for item in normalized:
            value = generalize_reasoning_text(item["reasoning_skill"].get(field, ""))
            result.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", value)
                if sentence.strip()
            )
        return list(dict.fromkeys(result))

    def shared_guidance(field: str) -> str:
        candidates = sentence_candidates(field)
        if not candidates:
            return ""
        if len(normalized) == 1:
            return candidates[0]

        def score(sentence: str) -> tuple[float, int, int]:
            tokens = tokenize(sentence)
            cross_scores = []
            for item in normalized:
                other = tokenize(item["reasoning_skill"].get(field, ""))
                cross_scores.append(len(tokens & other) / max(1, len(tokens | other)))
            specificity_penalty = len(re.findall(r"\d|[\"']", sentence))
            return sum(cross_scores) / len(cross_scores), -specificity_penalty, -len(sentence)

        return max(candidates, key=score)

    base["reasoning_skill"] = {
        field: shared_guidance(field)
        or base["reasoning_skill"][field]
        for field in ("M", "B", "V")
    }
    total_n = sum(int(item["statistics"].get("N", 0)) for item in normalized)
    total_s = sum(int(item["statistics"].get("S", 0)) for item in normalized)
    smoothing = float(base["statistics"].get("lambda", 1.0))
    base["statistics"].update(
        {
            "N": total_n,
            "S": min(total_s, total_n),
            "rho": (min(total_s, total_n) + smoothing) / (total_n + 2 * smoothing),
        }
    )
    keywords = []
    merged_from = []
    for item in normalized:
        keywords.extend(_listify(item["scenario_descriptor"]["W_k"].get("semantic_keywords")))
        if item.get("skill_id") != base.get("skill_id"):
            merged_from.append(item.get("skill_id"))
        merged_from.extend(_listify(item.get("merged_from")))
    base["scenario_descriptor"]["W_k"]["semantic_keywords"] = list(
        dict.fromkeys(keyword for keyword in keywords if keyword)
    )[:30]
    base["merged_from"] = list(dict.fromkeys(value for value in merged_from if value))
    return base


def semantic_compatibility(objective: str, context: PageContext, w_k: Mapping[str, Any]) -> float:
    """Rank a candidate using its compact semantic descriptor ``W_k``."""

    objective_tokens = tokenize(objective)
    page_tokens = set(context.ui_tokens)
    keyword_tokens = tokenize(w_k.get("semantic_keywords", []))
    intent_tokens = tokenize(
        " ".join(
            _listify(w_k.get("task_intent"))
            + _listify(w_k.get("task_family"))
            + _listify(w_k.get("scenario_description"))
        )
    )
    descriptor_tokens = keyword_tokens | intent_tokens
    if not descriptor_tokens:
        return 0.0

    objective_overlap = len(objective_tokens & descriptor_tokens) / math.sqrt(
        max(1, len(objective_tokens) * len(descriptor_tokens))
    )
    keyword_coverage = len(objective_tokens & keyword_tokens) / max(1, len(keyword_tokens))
    page_overlap = len(page_tokens & descriptor_tokens) / math.sqrt(
        max(1, len(page_tokens) * len(descriptor_tokens))
    )
    phrase_bonus = 0.0
    objective_lower = objective.lower()
    for phrase in _listify(w_k.get("semantic_keywords")):
        if len(phrase) >= 4 and phrase.lower() in objective_lower:
            phrase_bonus = min(0.2, phrase_bonus + 0.05)
    score = min(
        1.0,
        0.62 * objective_overlap
        + 0.23 * keyword_coverage
        + 0.15 * page_overlap
        + phrase_bonus,
    )
    objective_actions = _action_groups(objective_tokens)
    descriptor_actions = _action_groups(descriptor_tokens)
    multiplier = 1.0
    bonus = 0.0
    if objective_actions and descriptor_actions and objective_actions.isdisjoint(descriptor_actions):
        multiplier *= 0.03
    elif objective_actions and not descriptor_actions:
        multiplier *= 0.65
    elif objective_actions & descriptor_actions:
        bonus += 0.06
        if descriptor_actions - objective_actions:
            multiplier *= 0.25

    objective_objects = _object_groups(objective_tokens)
    descriptor_objects = _object_groups(descriptor_tokens)
    if objective_objects and descriptor_objects and objective_objects.isdisjoint(descriptor_objects):
        multiplier *= 0.12
    elif objective_objects & descriptor_objects:
        bonus += 0.06
        if descriptor_objects - objective_objects:
            multiplier *= 0.6

    polarity_pairs = (("upvote", "downvote"), ("subscribe", "unsubscribe"))
    for wanted, opposite in polarity_pairs:
        if wanted in objective_tokens and opposite in descriptor_tokens:
            multiplier *= 0.05
        if opposite in objective_tokens and wanted in descriptor_tokens:
            multiplier *= 0.05
    return min(1.0, (score + bonus) * multiplier)


@dataclass
class RetrievalCandidate:
    skill: Any
    descriptor: dict[str, dict[str, Any]]
    score: float


@dataclass
class DualSkillSelection:
    """Singleton activation result for one environment step."""

    context: PageContext
    interaction: Optional[RetrievalCandidate] = None
    reasoning: Optional[RetrievalCandidate] = None
    interaction_candidates: list[RetrievalCandidate] = field(default_factory=list)
    reasoning_candidates: list[RetrievalCandidate] = field(default_factory=list)
    current_intent: str = ""
    contract_mode: str = "drive"
    shared_slot_id: Optional[str] = None


class ScenarioAwareCoordinator:
    """Two-stage retrieval followed by per-level singleton activation."""

    def __init__(
        self,
        registry: Any,
        *,
        interaction_threshold: float = 0.06,
        reasoning_threshold: float = 0.06,
        use_interaction_skills: bool = True,
        use_reasoning_skills: bool = True,
        contract_mode: str = "drive",
    ) -> None:
        self.registry = registry
        self.interaction_threshold = interaction_threshold
        self.reasoning_threshold = reasoning_threshold
        self.use_interaction_skills = use_interaction_skills
        self.use_reasoning_skills = use_reasoning_skills
        if contract_mode not in {"shared_contract", "joint_typed", "drive"}:
            raise ValueError(f"Unknown reuse contract mode: {contract_mode}")
        self.contract_mode = contract_mode
        self.last_selection: Optional[DualSkillSelection] = None

    @staticmethod
    def _rank(candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.score,
                float(getattr(item.skill, "metadata", {}).get("utility_score", 0.5))
                if not isinstance(item.skill, Mapping)
                else float(item.skill.get("statistics", {}).get("rho", 0.5)),
                getattr(item.skill, "name", "")
                if not isinstance(item.skill, Mapping)
                else str(item.skill.get("skill_id", "")),
            ),
            reverse=True,
        )

    def retrieve_reasoning(
        self,
        objective: str,
        context: PageContext,
    ) -> tuple[Optional[RetrievalCandidate], list[RetrievalCandidate]]:
        """Paper Algorithm 1, lines 3--4: retrieve one reasoning skill first."""

        candidates: list[RetrievalCandidate] = []
        library = (
            getattr(self.registry, "external_task_lessons", [])
            if self.use_reasoning_skills
            else []
        )
        for raw_lesson in library:
            lesson = normalize_reasoning_skill(
                raw_lesson, site_hint=getattr(self.registry, "site_name", "")
            )
            descriptor = lesson["scenario_descriptor"]
            if not structural_match(context, descriptor["U_k"]):
                continue
            score = semantic_compatibility(objective, context, descriptor["W_k"])
            if score >= self.reasoning_threshold:
                candidates.append(RetrievalCandidate(lesson, descriptor, score))
        ranked = self._rank(candidates)
        return (ranked[0] if ranked else None), ranked

    def retrieve_interaction(
        self,
        objective: str,
        current_intent: str,
        context: PageContext,
    ) -> tuple[Optional[RetrievalCandidate], list[RetrievalCandidate]]:
        """Algorithm 1, lines 7--9: retrieve after ``g_t`` has been formed."""

        candidates: list[RetrievalCandidate] = []
        query = " ".join(value for value in (objective, current_intent) if value).strip()
        for skill in self.registry.get_all_skills() if self.use_interaction_skills else []:
            descriptor = interaction_descriptor(
                skill, site_hint=getattr(self.registry, "site_name", "")
            )
            if not structural_match(context, descriptor["U_k"]):
                continue
            score = semantic_compatibility(query, context, descriptor["W_k"])
            if score >= self.interaction_threshold:
                candidates.append(RetrievalCandidate(skill, descriptor, score))
        ranked = self._rank(candidates)
        return (ranked[0] if ranked else None), ranked

    def select(
        self,
        objective: str,
        url: str,
        observation: Any,
        *,
        current_intent: Optional[str] = None,
        step: Optional[int] = None,
        reasoning_stage: Optional[
            tuple[PageContext, Optional[RetrievalCandidate], Sequence[RetrievalCandidate]]
        ] = None,
    ) -> DualSkillSelection:
        if reasoning_stage is None:
            context = extract_context(
                url, observation, site_hint=getattr(self.registry, "site_name", "")
            )
            selected_reasoning, reasoning_ranked = self.retrieve_reasoning(objective, context)
        else:
            # The actor already executed Algorithm 1's reasoning stage to
            # form g_t.  Reuse that exact result rather than re-retrieving a
            # potentially different rule before interaction retrieval.
            context, selected_reasoning, reasoning_ranked = reasoning_stage
            reasoning_ranked = list(reasoning_ranked)
        # Compatibility callers that do not provide g_t use q as the intent.  The
        # maintained Agent path always supplies an explicitly formed intent.
        formed_intent = (current_intent or objective).strip()
        interaction_query = " ".join(
            value for value in (objective, formed_intent) if value
        ).strip()
        selected_interaction, interaction_ranked = self.retrieve_interaction(
            objective, formed_intent, context
        )
        shared_slot_id = None

        if (
            self.contract_mode == "shared_contract"
            and self.use_interaction_skills
            and self.use_reasoning_skills
        ):
            # Freeze one equal-information joint corpus by deterministically
            # pairing the complete typed libraries. Shared-contract retrieval
            # must consume both fields from one slot; typed modes may retrieve
            # the same fields independently. Cycling the shorter library keeps
            # every source entry represented without adding information.
            raw_i = sorted(
                self.registry.get_all_skills(), key=lambda skill: getattr(skill, "name", "")
            )
            raw_r = sorted(
                (
                    normalize_reasoning_skill(item, site_hint=getattr(self.registry, "site_name", ""))
                    for item in getattr(self.registry, "external_task_lessons", [])
                ),
                key=lambda item: str(item.get("skill_id", item.get("cluster_id", ""))),
            )
            slot_count = max(len(raw_i), len(raw_r)) if raw_i and raw_r else 0
            slots = []
            for index in range(slot_count):
                i_skill = raw_i[index % len(raw_i)]
                r_skill = raw_r[index % len(raw_r)]
                i_descriptor = interaction_descriptor(
                    i_skill, site_hint=getattr(self.registry, "site_name", "")
                )
                r_descriptor = r_skill["scenario_descriptor"]
                if not structural_match(context, i_descriptor["U_k"]):
                    continue
                if not structural_match(context, r_descriptor["U_k"]):
                    continue
                # Even the shared-corpus ablation preserves DRIVE's temporal
                # ordering: the interaction side is ranked only after the
                # current intent has been formed from the retrieved reasoning.
                i_score = semantic_compatibility(interaction_query, context, i_descriptor["W_k"])
                r_score = semantic_compatibility(objective, context, r_descriptor["W_k"])
                i_item = RetrievalCandidate(i_skill, i_descriptor, i_score)
                r_item = RetrievalCandidate(r_skill, r_descriptor, r_score)
                # One shared descriptor must trade off semantic compatibility
                # of both fields rather than invoking two typed retrievers.
                shared_score = (i_item.score + r_item.score) / 2.0
                if shared_score < min(self.interaction_threshold, self.reasoning_threshold):
                    continue
                slots.append((shared_score, index, i_item, r_item))
            if slots:
                _, slot_index, selected_interaction, selected_reasoning = max(
                    slots, key=lambda item: (item[0], -item[1])
                )
                shared_slot_id = f"shared_slot_{slot_index:03d}"
            else:
                selected_interaction = None
                selected_reasoning = None
        selection = DualSkillSelection(
            context=context,
            interaction=selected_interaction,
            reasoning=selected_reasoning,
            interaction_candidates=interaction_ranked,
            reasoning_candidates=reasoning_ranked,
            current_intent=formed_intent,
            contract_mode=self.contract_mode,
            shared_slot_id=shared_slot_id,
        )
        self.last_selection = selection

        if hasattr(self.registry, "set_active_selection"):
            self.registry.set_active_selection(selection, objective=objective, step=step)
        return selection


def reasoning_similarity(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Similarity used by batch reasoning-skill consolidation."""

    first_n = normalize_reasoning_skill(first)
    second_n = normalize_reasoning_skill(second)
    first_tokens = tokenize(
        list(first_n["reasoning_skill"].values())
        + list(first_n["scenario_descriptor"]["W_k"].values())
    )
    second_tokens = tokenize(
        list(second_n["reasoning_skill"].values())
        + list(second_n["scenario_descriptor"]["W_k"].values())
    )
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def descriptor_json_path(code_path: Path) -> Path:
    """Return the conventional metadata path for a direct skill file."""

    return code_path.with_suffix(".json")
