# Grok-powered natural language → UserFilterPrefs parser.
#
# NLP parsing is available for ALL filter fields, not just tones.
# The UI may present shortcut chips or dropdowns as convenience suggestions,
# but the backend accepts free-text descriptions for any setting.
#
# What gets parsed to what:
#   Hard enum fields    → resolved to integer values (exact match downstream)
#   Soft enum (tone)    → resolved to integers PLUS raw description stored
#                         for dense embedding (richer soft matching)
#   Free fields (topic) → kept as strings, embedded at save time
#   Account size        → enum tier OR numeric range (e.g. "under 50k followers")
#   Language settings   → parsed into the three-way translation model

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Lookup tables ─────────────────────────────────────────────────────────────

STYLE_MAP = {
    "formal":0,"academic":0,"scientific":0,"corporate":0,"professional":0,
    "legal":0,"bureaucratic":0,"journalistic":0,"political":0,"official":0,
    "informal":1,"casual":1,"colloquial":1,"slang":1,"dialect":1,
    "simple":1,"poetic":1,"conversational":1,
    "vulgar":2,"profane":2,"swearing":2,"insulting":2,"discriminatory":2,
    "offensive":2,
}

TONE_MAP = {
    "objective":0,"neutral":1,"analytical":2,"robotic":3,"sensationalist":4,
    "enthusiastic":5,"friendly":6,"sympathetic":7,"diplomatic":8,"passionate":9,
    "aggressive":10,"angry":11,"annoyed":12,"confident":13,"humorous":14,
    "ironic":15,"polemic":16,"persuasive":17,"urgent":18,"surprised":19,
    "apologetic":20,"sad":21,"skeptical":22,"condescending":23,"outraged":24,
    # synonyms
    "sarcastic":15,"funny":14,"cynical":22,"critical":22,"reasoned":2,
    "thoughtful":2,"measured":1,"balanced":1,"inflammatory":24,
    "preachy":16,"calm":1,"dry":2,"emotional":9,
}

CLAIM_MAP = {
    "fact":0,"fact_claim":0,"factual":0,"objective":0,"informative":0,
    "opinion":1,"subjective":1,"opinion_based":1,"personal":1,
    "mixed":2,
}

VERIFICATION_MAP = {
    "none":0,"unverified":0,
    "community":1,"community_note":1,"community_noted":1,
    "ai":2,"ai_checked":2,"grok":2,"ai_verified":2,
    "manual":3,"manually_checked":3,"editorial":3,
}

SENTIMENT_MAP = {
    "positive":0,"supportive":0,"optimistic":0,"constructive":0,
    "neutral":1,
    "negative":2,"critical":2,"pessimistic":2,
    "ambivalent":3,"mixed":3,
}

TIER_MAP = {
    "nano":0,"tiny":0,"micro":0,
    "small":1,"indie":1,"independent":1,
    "medium":2,
    "large":3,"big":3,
    "mega":4,"huge":4,"viral":4,"mainstream":4,"celebrity":4,
}

BIAS_MAP = {
    "interest":0,"interest_driven":0,"topical":0,"semantic":0,
    "social":1,"social_driven":1,"trending":1,"engagement":1,
    "discovery":2,"discovery_driven":2,"explore":2,
    "hidden gems":2,"under the radar":2,"niche":2,
}

SIGNAL_MAP = {
    "like":0,"likes":0,
    "reply":1,"replies":1,"comments":1,
    "repost":2,"reposts":2,"retweet":2,
    "post":3,"posts":3,"own posts":3,
    "dwell":4,"reading time":4,"time spent":4,
    "click":5,"clicks":5,
}

POST_TYPE_MAP = {
    "original":0,"new post":0,"posts":0,
    "repost":1,"reposts":1,"retweets":1,
    "reply":2,"replies":2,"comments":2,
    "quote":3,"quote tweet":3,"quotes":3,
}

GRAPH_MAP = {
    "following_only":0,"following":0,"only following":0,
    "friends_of_friends":1,"fof":1,"friends of friends":1,
    "network_extended":2,"extended":2,"extended network":2,
    "global":3,"everyone":3,"all":3,
}


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are a feed filter configuration assistant for a social media recommendation system.
Parse the user's natural-language filter request and return ONLY a valid JSON object.
Do not include markdown fences or any text outside the JSON.

IMPORTANT: Parse the user's intent as specifically as they state it — do NOT generalise.
"posts about LoRA fine-tuning" → preferred_topics: ["LoRA fine-tuning"], not ["AI"].

Output schema (all fields optional — omit rather than null or empty list):
{{
  "graph_depth": str,               // "following_only"|"friends_of_friends"|"network_extended"|"global"
  "content_max_age_days": int,

  // Language settings
  "included_original_languages": [str],  // ISO 639-1 codes e.g. ["en","de"]
  "show_auto_translated": bool,          // default true
  "show_manually_translated": bool,      // default true
  "accepted_translation_targets": [str], // target languages for translated content

  // Style (hard enum, whitelist/blacklist)
  "included_styles": [str],   // whitelist: only show these; empty=show all
  "excluded_styles": [str],   // blacklist: never show these

  // Tone (soft enum + optional dense embedding)
  "included_tones": [str],    // whitelist: only posts with at least one of these
  "suppressed_tones": [str],  // soft penalise
  "preferred_tones": [str],   // score boost
  "raw_tone_description": str, // verbatim phrase for dense embedding e.g. "analytical but not dry"

  // Topics (free field — kept as strings for embedding)
  "preferred_topics": [str],
  "blocked_topics": [str],

  // Account size: enum tiers OR numeric range (use one, not both)
  "included_tiers": [str],    // "nano"|"small"|"medium"|"large"|"mega"
  "min_followers": int,
  "max_followers": int,

  // Claim type and verification (hard enum, whitelist)
  "included_claim_types": [str],      // "fact"|"opinion"|"mixed"
  "required_verification": [str],     // "none"|"community"|"ai"|"manual"

  // Sentiment (hard enum, whitelist)
  "included_sentiments": [str],       // "positive"|"neutral"|"negative"|"ambivalent"

  // Post types shown in feed (whitelist)
  "included_post_types": [str],       // "original"|"repost"|"reply"|"quote"

  // Recommendation signal preferences
  "included_signals": [str],          // "like"|"reply"|"repost"|"post"|"dwell"|"click"
  "signal_weights": {{"like": float, "reply": float, "repost": float, "post": float}},
  "history_lookback_days": int,

  // Presentation
  "bias": str,                        // "interest"|"social"|"discovery"
  "boost_low_impression_content": bool,
  "engagement_vs_impression_ratio": float,  // 0.0=count, 1.0=rate

  // Social proof
  "social_proof_min_overlap": int,

  // Safety
  "show_illicit_targeting_self": bool
}}

Mapping rules:
- "corporate","academic","legal","professional" → style "formal"
- "casual","slang","dialect" → style "informal"
- "swearing","insulting","discriminatory","profane" → style "vulgar"
- Tone synonyms: "sarcastic"→"ironic","funny"→"humorous","reasoned"→"analytical",
  "thoughtful"→"analytical","preachy"→"polemic","inflammatory"→"outraged","cynical"→"skeptical"
- "independent creators","small accounts" → included_tiers: ["nano","small"]
- "hidden gems","under the radar","niche" → bias: "discovery", boost_low_impression_content: true
- "only original posts","no reposts","no replies" → set included_post_types accordingly
- "last year","from 2023" → content_max_age_days using days from that period
- "under 50k followers" → max_followers: 50000 (use numeric range, not tiers)
- "based on my likes only" → included_signals: ["like"]
- If the user expresses a nuanced tone quality (e.g. "not too dry","engaging but calm"),
  include the raw phrase verbatim in raw_tone_description in addition to resolved enum values.
  This phrase will be embedded as a dense vector for richer soft tone matching.
- Fact-checked = required_verification: ["community","ai","manual"] (any verification)
- "no reposts in my feed" → included_post_types: ["original","reply","quote"]
  (this controls what appears, not what signals are used for recommendation)
- "don't use my reposts for recommendations" → excluded from included_signals
""".strip()


# ── Parser ────────────────────────────────────────────────────────────────────

class IntentParser:
    """
    Parses any natural-language filter description into structured prefs fields.

    All fields — not just tones — are parseable via natural language.
    The client-side shortcut UI (chips, dropdowns) populates the same fields;
    the backend is indifferent to how the user expressed their preference.

    Workflow (called at preference-SAVE time, not hot path):
      1. Call Grok with user description
      2. Parse JSON response
      3. Resolve enum strings → integers
      4. Return structured dict
      5. Caller embeds topic strings + raw_tone_description (PreferenceStore.save)
    """

    def __init__(self, grok_client: Optional[Any] = None) -> None:
        self._client = grok_client

    def parse(self, description: str) -> dict:
        raw = self._call_grok(description)
        parsed = self._parse_json(raw)
        resolved = self._resolve(parsed)
        resolved["grok_intent_description"] = description
        return resolved

    def _call_grok(self, description: str) -> str:
        if self._client is None:
            raise NotImplementedError(
                "IntentParser requires a Grok client.\n"
                "Expected: client.complete(system: str, user: str) -> str"
            )
        return self._client.complete(system=SYSTEM_PROMPT, user=description)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        s = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            result = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Grok returned non-JSON: {raw!r}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Expected object, got {type(result).__name__}")
        return result

    def _resolve(self, raw: dict) -> dict:
        out: dict = {}

        # ── Scalars ───────────────────────────────────────────────────────
        for k in ("content_max_age_days","history_lookback_days",
                  "social_proof_min_overlap","min_followers","max_followers"):
            if (v := raw.get(k)) is not None:
                out[k] = int(v)

        for k in ("engagement_vs_impression_ratio",):
            if (v := raw.get(k)) is not None:
                out[k] = float(v)

        for k in ("show_auto_translated","show_manually_translated",
                  "boost_low_impression_content","show_illicit_targeting_self"):
            if (v := raw.get(k)) is not None:
                out[k] = bool(v)

        # ── Free strings ──────────────────────────────────────────────────
        for k in ("raw_tone_description","grok_intent_description"):
            if v := raw.get(k):
                out[k] = str(v)

        # ── Language (strings, not enum) ──────────────────────────────────
        for k in ("included_original_languages","accepted_translation_targets"):
            if items := raw.get(k):
                out[k] = [str(i).lower() for i in items]

        # ── Hard enum lists ───────────────────────────────────────────────
        out |= self._resolve_list("included_styles",   raw, STYLE_MAP)
        out |= self._resolve_list("excluded_styles",   raw, STYLE_MAP)
        out |= self._resolve_list("included_claim_types", raw, CLAIM_MAP)
        out |= self._resolve_list("required_verification", raw, VERIFICATION_MAP)
        out |= self._resolve_list("included_sentiments", raw, SENTIMENT_MAP)
        out |= self._resolve_list("included_post_types", raw, POST_TYPE_MAP)
        out |= self._resolve_list("included_signals",  raw, SIGNAL_MAP)
        out |= self._resolve_list("included_tiers",    raw, TIER_MAP)

        # ── Soft enum lists (tone) ────────────────────────────────────────
        out |= self._resolve_list("included_tones",   raw, TONE_MAP)
        out |= self._resolve_list("suppressed_tones", raw, TONE_MAP)
        out |= self._resolve_list("preferred_tones",  raw, TONE_MAP)

        # ── Free-field lists (topics — kept as strings for embedding) ─────
        for k in ("preferred_topics","blocked_topics"):
            if items := raw.get(k):
                out[k] = [str(t) for t in items]

        # ── Graph depth ───────────────────────────────────────────────────
        if gd := raw.get("graph_depth"):
            if (v := GRAPH_MAP.get(gd.lower())) is not None:
                out["graph_depth"] = v
            else:
                logger.warning("Unknown graph_depth %r", gd)

        # ── Bias ──────────────────────────────────────────────────────────
        if b := raw.get("bias"):
            if (v := BIAS_MAP.get(b.lower())) is not None:
                out["bias"] = v

        # ── Signal weights ────────────────────────────────────────────────
        if weights := raw.get("signal_weights"):
            out["signal_weights"] = {
                SIGNAL_MAP[k.lower()]: float(v)
                for k, v in weights.items()
                if k.lower() in SIGNAL_MAP
            }

        return out

    @staticmethod
    def _resolve_list(key: str, raw: dict, lookup: dict) -> dict:
        items = raw.get(key)
        if not items:
            return {}
        resolved, seen = [], set()
        for item in items:
            v = lookup.get(str(item).lower())
            if v is not None and v not in seen:
                resolved.append(v)
                seen.add(v)
            elif v is None:
                logger.warning("Unknown %r value %r — skipping", key, item)
        return {key: resolved} if resolved else {}


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EXAMPLES = {
        # Tests: tone (soft enum + raw desc), topics (free string), account size (custom range)
        "Show me analytical posts about AI safety, nothing corporate or outraged, "
        "prefer independent researchers under 50k followers, based on what I liked in 2023":
        """{
            "included_styles": ["informal"],
            "excluded_styles": ["formal"],
            "preferred_tones": ["analytical","skeptical"],
            "suppressed_tones": ["outraged"],
            "raw_tone_description": "analytical but not preachy",
            "preferred_topics": ["AI safety","AI alignment"],
            "max_followers": 50000,
            "included_claim_types": ["fact"],
            "history_lookback_days": 365,
            "included_signals": ["like"]
        }""",

        # Tests: language filter with translation nuance
        "Show me posts in English and German. I'm fine with Spanish posts if manually "
        "translated to English by the author, but no platform auto-translations":
        """{
            "included_original_languages": ["en","de"],
            "show_auto_translated": false,
            "show_manually_translated": true,
            "accepted_translation_targets": ["en","de"]
        }""",

        # Tests: post type filter (feed content) vs signal filter (recommendations)
        "I only want original posts and quote tweets in my feed, no reposts or replies. "
        "But still use my reposts to build my interest profile":
        """{
            "included_post_types": ["original","quote"],
            "included_signals": ["like","reply","repost","post","dwell"]
        }""",

        # Tests: verification filter, community notes
        "Only show me fact-checked content — either by community notes or AI":
        """{
            "required_verification": ["community","ai"],
            "included_claim_types": ["fact","mixed"]
        }""",

        # Tests: discovery bias + account size tiers
        "Hidden gems: niche creators, low follower accounts, posts I haven't seen yet":
        """{
            "bias": "discovery",
            "boost_low_impression_content": true,
            "included_tiers": ["nano","small"]
        }""",
    }

    class StubGrok:
        def __init__(self, m): self._m = m
        def complete(self, system, user): return self._m.get(user, "{}")

    parser = IntentParser(grok_client=StubGrok(EXAMPLES))

    for desc, _ in EXAMPLES.items():
        print(f"INPUT:  {desc[:80]}{'...' if len(desc)>80 else ''}")
        result = parser.parse(desc)
        # Show the key distinctions
        notes = []
        if "preferred_topics" in result:
            notes.append(f"topics kept as strings (will be embedded): {result['preferred_topics']}")
        if "preferred_tones" in result:
            notes.append(f"tones as enum ints (soft match): {result['preferred_tones']}")
        if "raw_tone_description" in result:
            notes.append(f"raw tone desc (will be embedded): '{result['raw_tone_description']}'")
        if "max_followers" in result and "included_tiers" not in result:
            notes.append(f"account size as numeric range: max={result['max_followers']}")
        if "included_tiers" in result:
            notes.append(f"account size as tiers: {result['included_tiers']}")
        if "included_post_types" in result:
            notes.append(f"feed post types (whitelist): {result['included_post_types']}")
        if "included_signals" in result:
            notes.append(f"recommendation signals (whitelist): {result['included_signals']}")

        print(f"OUTPUT: {json.dumps({k:v for k,v in result.items() if k != 'grok_intent_description'}, indent=2)}")
        for n in notes:
            print(f"  ↳ {n}")
        print()