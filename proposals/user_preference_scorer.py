# Sample scoring and filtering logic demonstrating the preference model.
#
# Fits into x-algorithm as:
#   - passes_hard_filters()  → home-mixer/filters/content_quality_filter.rs
#   - score_candidate()      → home-mixer/scorers/user_preference_scorer.rs
#   - FeedContextDefaults    → home-mixer/candidate_pipeline/feed_context.rs
#   - PreferenceStore        → called at save time, not hot path

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np

# ── Enums ─────────────────────────────────────────────────────────────────────

class SocialGraphDepth(IntEnum):
    FOLLOWING_ONLY=0; FOLLOWERS_OF_FOLLOWERS=1; NETWORK_EXTENDED=2; GLOBAL=3

    def default_threshold(self) -> float:
        return {0: 0.25, 1: 0.50, 2: 0.75, 3: 0.95}[int(self)]

class ContentStyleGroup(IntEnum):
    FORMAL=0; INFORMAL=1; VULGAR=2
    # ILLICIT enforced server-side only

class ContentTone(IntEnum):
    OBJECTIVE=0;  NEUTRAL=1;      ANALYTICAL=2;   ROBOTIC=3
    SENSATIONALIST=4; ENTHUSIASTIC=5; FRIENDLY=6; SYMPATHETIC=7
    DIPLOMATIC=8; PASSIONATE=9;   AGGRESSIVE=10;  ANGRY=11
    ANNOYED=12;   CONFIDENT=13;   HUMOROUS=14;    IRONIC=15
    POLEMIC=16;   PERSUASIVE=17;  URGENT=18;      SURPRISED=19
    APOLOGETIC=20; SAD=21;        SKEPTICAL=22;   CONDESCENDING=23
    OUTRAGED=24

class ClaimType(IntEnum):
    FACT=0; OPINION=1; MIXED=2

class VerificationStatus(IntEnum):
    NONE=0; COMMUNITY=1; AI_CHECKED=2; MANUAL=3

class Sentiment(IntEnum):
    POSITIVE=0; NEUTRAL=1; NEGATIVE=2; AMBIVALENT=3

class AccountSizeTier(IntEnum):
    NANO=0; SMALL=1; MEDIUM=2; LARGE=3; MEGA=4

    @classmethod
    def from_count(cls, n: int) -> "AccountSizeTier":
        if n < 1_000:      return cls.NANO
        if n < 10_000:     return cls.SMALL
        if n < 100_000:    return cls.MEDIUM
        if n < 1_000_000:  return cls.LARGE
        return cls.MEGA

class RankingBias(IntEnum):
    INTEREST=0; SOCIAL=1; DISCOVERY=2

class PostType(IntEnum):
    ORIGINAL=0; REPOST=1; REPLY=2; QUOTE=3

class FeedContext(IntEnum):
    FOR_YOU=0; FOLLOWING=1; TOPIC=2; EXPLORE=3; SEARCH=4; PROFILE=5

class EngagementSignal(IntEnum):
    LIKE=0; REPLY=1; REPOST=2; POST=3; DWELL=4; CLICK=5

    def default_weight(self) -> float:
        return {0:1.0, 1:2.0, 2:3.0, 3:5.0, 4:0.5, 5:0.5}[int(self)]

EMBEDDING_DIM = 64  # matches Phoenix post encoder dimension
GRAVITY = 1.8       # time-decay exponent


# ── Post features ──────────────────────────────────────────────────────────────

@dataclass
class PostFeatures:
    post_id:             int
    base_score:          float        # WeightedScorer output
    created_at_unix:     int
    post_type:           PostType
    is_illicit:          bool         # server-only tag
    illicit_targets_user_id: Optional[int]  # which user this illicit content targets

    # Hard-enum fields
    style_group:         ContentStyleGroup
    tones:               list[ContentTone]
    sentiment:           Sentiment
    claim_type:          ClaimType
    verification_statuses: list[VerificationStatus]

    # Language
    original_language:   str          # ISO 639-1
    available_translation_targets: list[str]   # languages with translations
    has_manual_translation: bool      # author provided translation
    has_auto_translation:   bool      # platform auto-translated

    # Embedding fields (from Phoenix two-tower encoder)
    post_embedding:      np.ndarray   # [D] general
    tone_embedding:      np.ndarray   # [D] tone-specific

    # Social proof
    seen_by_n_followed:  int

    # Engagement metrics
    impression_count:    int
    author_followers:    int

    # Topics (tags from NLP pipeline)
    topic_ids:           list[str]


@dataclass
class UserInteraction:
    signal:         EngagementSignal
    timestamp_unix: int
    topic_ids:      list[str]


# ── Feed context defaults ──────────────────────────────────────────────────────
#
# Each feed context has defaults that apply when the user has not set a
# preference. UserFilterPrefs overrides any default.

@dataclass
class FeedDefaults:
    graph_depth:           SocialGraphDepth
    included_post_types:   list[PostType]    # empty = all
    bias:                  RankingBias
    boost_low_impression:  bool
    apply_interest_filter: bool              # Following feed skips interest filter
    social_proof_on:       bool

FEED_DEFAULTS: dict[FeedContext, FeedDefaults] = {
    FeedContext.FOR_YOU: FeedDefaults(
        graph_depth=SocialGraphDepth.FOLLOWERS_OF_FOLLOWERS,
        included_post_types=[],  # all
        bias=RankingBias.INTEREST,
        boost_low_impression=False,
        apply_interest_filter=True,
        social_proof_on=True,
    ),
    FeedContext.FOLLOWING: FeedDefaults(
        graph_depth=SocialGraphDepth.FOLLOWING_ONLY,
        included_post_types=[],  # all — user controls via prefs
        bias=RankingBias.SOCIAL,
        boost_low_impression=False,
        apply_interest_filter=False,  # chronological intent; no interest filter
        social_proof_on=False,
    ),
    FeedContext.TOPIC: FeedDefaults(
        graph_depth=SocialGraphDepth.GLOBAL,
        included_post_types=[PostType.ORIGINAL, PostType.QUOTE],
        bias=RankingBias.INTEREST,
        boost_low_impression=False,
        apply_interest_filter=True,
        social_proof_on=True,
    ),
    FeedContext.EXPLORE: FeedDefaults(
        graph_depth=SocialGraphDepth.GLOBAL,
        included_post_types=[PostType.ORIGINAL, PostType.QUOTE],
        bias=RankingBias.DISCOVERY,
        boost_low_impression=True,
        apply_interest_filter=True,
        social_proof_on=False,
    ),
    FeedContext.SEARCH: FeedDefaults(
        graph_depth=SocialGraphDepth.GLOBAL,
        included_post_types=[],
        bias=RankingBias.SOCIAL,
        boost_low_impression=False,
        apply_interest_filter=True,
        social_proof_on=False,
    ),
    FeedContext.PROFILE: FeedDefaults(
        graph_depth=SocialGraphDepth.FOLLOWING_ONLY,
        included_post_types=[],
        bias=RankingBias.SOCIAL,
        boost_low_impression=False,
        apply_interest_filter=False,
        social_proof_on=False,
    ),
}


# ── User filter prefs ──────────────────────────────────────────────────────────

@dataclass
class LanguagePrefs:
    included_original_languages:  list[str] = field(default_factory=list)  # empty=all
    show_auto_translated:          bool      = True
    show_manually_translated:      bool      = True
    accepted_translation_targets:  list[str] = field(default_factory=list)  # empty=same as included_original

@dataclass
class StylePrefs:
    included_styles: list[ContentStyleGroup] = field(default_factory=list)  # empty=all
    excluded_styles: list[ContentStyleGroup] = field(default_factory=list)

@dataclass
class TonePrefs:
    included_tones:  list[ContentTone] = field(default_factory=list)  # empty=all
    suppressed_tones: list[ContentTone] = field(default_factory=list)
    preferred_tones: list[ContentTone] = field(default_factory=list)
    boost_weight:    float = 0.30
    suppress_weight: float = 0.50
    # Dense embedding of the user's raw tone description, pre-computed at save time
    tone_query_embedding: Optional[np.ndarray] = None

@dataclass
class TopicPrefs:
    preferred_embeddings: list[np.ndarray] = field(default_factory=list)
    blocked_embeddings:   list[np.ndarray] = field(default_factory=list)
    boost_weight:    float = 0.40
    block_strength:  float = 0.80  # 0=soft, 1=hard exclude

@dataclass
class AccountSizePrefs:
    included_tiers:  list[AccountSizeTier] = field(default_factory=list)  # empty=all
    min_followers:   Optional[int]         = None  # custom range
    max_followers:   Optional[int]         = None

    def matches(self, follower_count: int) -> bool:
        """True if account is within ANY of the specified constraints."""
        tier = AccountSizeTier.from_count(follower_count)
        tier_match = not self.included_tiers or tier in self.included_tiers
        if self.min_followers is not None or self.max_followers is not None:
            lo = self.min_followers or 0
            hi = self.max_followers or 10**15
            range_match = lo <= follower_count <= hi
            # Numeric range overrides tiers when explicitly set
            return range_match
        return tier_match

@dataclass
class ClaimPrefs:
    included_claim_types:     list[ClaimType]          = field(default_factory=list)
    required_verification:    list[VerificationStatus]  = field(default_factory=list)

@dataclass
class SignalPrefs:
    included_signals: list[EngagementSignal] = field(
        default_factory=lambda: list(EngagementSignal))
    signal_weights:   dict[EngagementSignal, float] = field(default_factory=dict)
    history_window_days: int = 30

    def effective_weight(self, signal: EngagementSignal) -> float:
        return self.signal_weights.get(signal, signal.default_weight())

@dataclass
class UserFilterPrefs:
    # Social graph
    graph_depth:               SocialGraphDepth = SocialGraphDepth.FOLLOWERS_OF_FOLLOWERS
    similarity_override:       Optional[float]  = None
    graph_similarity_overrides: dict[SocialGraphDepth, float] = field(default_factory=dict)

    # Content age
    content_max_age_days:      int = 7

    # Sub-filters — all optional; absent = use context defaults
    language:   LanguagePrefs     = field(default_factory=LanguagePrefs)
    style:      StylePrefs        = field(default_factory=StylePrefs)
    tone:       TonePrefs         = field(default_factory=TonePrefs)
    topics:     TopicPrefs        = field(default_factory=TopicPrefs)
    account_size: AccountSizePrefs = field(default_factory=AccountSizePrefs)
    claims:     ClaimPrefs        = field(default_factory=ClaimPrefs)

    # Whitelist of post types to show. Empty = use feed context default.
    included_post_types: list[PostType] = field(default_factory=list)

    # Whitelist of sentiments. Empty = all.
    included_sentiments: list[Sentiment] = field(default_factory=list)

    # Recommendation signals
    signals: SignalPrefs = field(default_factory=SignalPrefs)

    # Social proof
    social_proof_min_overlap:       int   = 3
    social_proof_relaxed_threshold: float = 0.15

    # Presentation
    bias:                     Optional[RankingBias] = None  # None = context default
    boost_low_impression:     Optional[bool]        = None
    low_impression_threshold: int                   = 5_000
    engagement_vs_impression_ratio: float           = 0.5

    # Illicit content: user may see illicit content targeting themselves
    show_illicit_targeting_self: bool = False

    def effective_similarity_threshold(self) -> float:
        return (
            self.similarity_override
            or self.graph_similarity_overrides.get(self.graph_depth)
            or self.graph_depth.default_threshold()
        )


# ── Hard filtering ─────────────────────────────────────────────────────────────
#
# Binary gates applied before scoring.
# Returns None (exclude) or the post (pass through).

def passes_hard_filters(
    post:     PostFeatures,
    prefs:    UserFilterPrefs,
    defaults: FeedDefaults,
    viewer_id: int,
) -> bool:

    # ── Illicit content ───────────────────────────────────────────────────
    # Always excluded unless it specifically targets the viewing user AND
    # that user has opted in to see it.
    if post.is_illicit:
        if prefs.show_illicit_targeting_self and post.illicit_targets_user_id == viewer_id:
            pass  # user opted in to see content targeting themselves
        else:
            return False

    # ── Post type ─────────────────────────────────────────────────────────
    # User pref takes precedence; fall back to feed context default.
    allowed_types = prefs.included_post_types or defaults.included_post_types
    if allowed_types and post.post_type not in allowed_types:
        return False

    # ── Content age ───────────────────────────────────────────────────────
    if (_unix_now() - post.created_at_unix) > prefs.content_max_age_days * 86_400:
        return False

    # ── Language ──────────────────────────────────────────────────────────
    lang = prefs.language
    in_original = (
        not lang.included_original_languages
        or post.original_language in lang.included_original_languages
    )
    if in_original:
        pass  # original language matches — post passes
    else:
        # Post is in a non-preferred original language — accept only via translation
        target_langs = lang.accepted_translation_targets or lang.included_original_languages
        has_acceptable_translation = any(
            t in target_langs for t in post.available_translation_targets
        )
        if not has_acceptable_translation:
            return False
        # Check translation type preference
        if post.has_auto_translation and not lang.show_auto_translated:
            if not (post.has_manual_translation and lang.show_manually_translated):
                return False
        if post.has_manual_translation and not lang.show_manually_translated:
            if not (post.has_auto_translation and lang.show_auto_translated):
                return False

    # ── Style (whitelist takes precedence over blacklist) ─────────────────
    if prefs.style.included_styles:
        if post.style_group not in prefs.style.included_styles:
            return False
    elif post.style_group in prefs.style.excluded_styles:
        return False

    # ── Tone whitelist (hard inclusion gate) ──────────────────────────────
    # If a tone whitelist is set, posts with NO matching tone are excluded.
    if prefs.tone.included_tones:
        if not any(t in prefs.tone.included_tones for t in post.tones):
            return False

    # ── Sentiment whitelist ───────────────────────────────────────────────
    if prefs.included_sentiments and post.sentiment not in prefs.included_sentiments:
        return False

    # ── Claim type whitelist ──────────────────────────────────────────────
    if prefs.claims.included_claim_types:
        if post.claim_type not in prefs.claims.included_claim_types:
            return False

    # ── Verification whitelist ────────────────────────────────────────────
    if prefs.claims.required_verification:
        if not any(v in prefs.claims.required_verification
                   for v in post.verification_statuses):
            return False

    # ── Account size ──────────────────────────────────────────────────────
    if (prefs.account_size.included_tiers
            or prefs.account_size.min_followers is not None
            or prefs.account_size.max_followers is not None):
        if not prefs.account_size.matches(post.author_followers):
            return False

    return True


# ── Soft scoring ───────────────────────────────────────────────────────────────

def _normalised_random(rows, cols, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((rows, cols)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m

TONE_EMBEDDINGS: np.ndarray = _normalised_random(len(ContentTone), EMBEDDING_DIM)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _tone_centroid(tones: list[ContentTone]) -> Optional[np.ndarray]:
    if not tones:
        return None
    vecs = TONE_EMBEDDINGS[[t.value for t in tones]]
    c = vecs.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else None


def tone_score(post: PostFeatures, prefs: TonePrefs) -> float:
    """
    Soft enum matching: cosine similarity against tone embedding centroids.
    Returns a signed delta (positive=boost, negative=suppress).

    When the user also provided a raw tone description, its pre-computed
    dense embedding is averaged with the enum centroid for richer matching.
    """
    if not post.tones:
        return 0.0
    post_tone_vec = _tone_centroid(post.tones)
    if post_tone_vec is None:
        return 0.0

    boost = 0.0
    if prefs.preferred_tones:
        pref_centroid = _tone_centroid(prefs.preferred_tones)
        sim = cosine(post_tone_vec, pref_centroid)
        enum_boost = ((sim + 1) / 2) * prefs.boost_weight
        # If a raw description embedding is available, average both signals
        if prefs.tone_query_embedding is not None:
            desc_sim = cosine(post_tone_vec, prefs.tone_query_embedding)
            desc_boost = ((desc_sim + 1) / 2) * prefs.boost_weight
            boost = (enum_boost + desc_boost) / 2
        else:
            boost = enum_boost

    suppress = 0.0
    if prefs.suppressed_tones:
        supp_centroid = _tone_centroid(prefs.suppressed_tones)
        sim = cosine(post_tone_vec, supp_centroid)
        suppress = ((sim + 1) / 2) * prefs.suppress_weight

    return boost - suppress


def topic_score(post: PostFeatures, prefs: TopicPrefs) -> tuple[float, bool]:
    """
    Embedding-based topic matching. No enum.
    Returns (score_delta, hard_blocked).
    """
    if not prefs.preferred_embeddings and not prefs.blocked_embeddings:
        return 0.0, False

    pv = post.post_embedding
    boost = 0.0
    if prefs.preferred_embeddings:
        sims = [cosine(pv, q) for q in prefs.preferred_embeddings]
        max_sim = max(sims)
        boost = ((max_sim + 1) / 2) * prefs.boost_weight

    if prefs.blocked_embeddings:
        sims = [cosine(pv, q) for q in prefs.blocked_embeddings]
        max_blocked = (max(sims) + 1) / 2
        if max_blocked * prefs.block_strength > 0.75:
            return 0.0, True  # hard exclude
        suppress = max_blocked * prefs.block_strength * 0.5
        return boost - suppress, False

    return boost, False


def interest_score(
    post:    PostFeatures,
    prefs:   SignalPrefs,
    history: list[UserInteraction],
) -> float:
    """
    Time-decayed affinity: Σ weight / (age_hours + 2)^γ
    Only interactions within history_window that share a topic are counted.
    """
    now = _unix_now()
    cutoff = now - prefs.history_window_days * 86_400
    post_topics = {t.lower() for t in post.topic_ids}
    score = 0.0
    for ev in history:
        if ev.timestamp_unix < cutoff:
            continue
        if ev.signal not in prefs.included_signals:
            continue
        if not {t.lower() for t in ev.topic_ids} & post_topics:
            continue
        age_h = (now - ev.timestamp_unix) / 3600.0
        score += prefs.effective_weight(ev.signal) / math.pow(age_h + 2.0, GRAVITY)
    return score


def social_proof_adjustment(
    post:  PostFeatures,
    prefs: UserFilterPrefs,
) -> tuple[float, float]:
    """Returns (effective_threshold, additive_boost)."""
    base = prefs.effective_similarity_threshold()
    if post.seen_by_n_followed >= prefs.social_proof_min_overlap:
        additive = 0.3 + 0.1 * min(post.seen_by_n_followed, 10)
        return min(base, prefs.social_proof_relaxed_threshold), additive
    return base, 0.0


def presentation_multiplier(
    post:     PostFeatures,
    prefs:    UserFilterPrefs,
    defaults: FeedDefaults,
) -> float:
    bias = prefs.bias if prefs.bias is not None else defaults.bias
    if bias == RankingBias.SOCIAL:
        rate = post.impression_count and (1 / post.impression_count * 10_000)
        w = prefs.engagement_vs_impression_ratio
        return max(0.1, (1 - w) + w * rate * 20)
    if bias == RankingBias.DISCOVERY:
        threshold = prefs.low_impression_threshold
        boost = prefs.boost_low_impression if prefs.boost_low_impression is not None \
                else defaults.boost_low_impression
        if post.impression_count < threshold or boost:
            return 2.0
        return 0.5
    return 1.0  # INTEREST: neutral multiplier


def score_candidate(
    post:      PostFeatures,
    prefs:     UserFilterPrefs,
    defaults:  FeedDefaults,
    history:   list[UserInteraction],
    viewer_id: int,
) -> Optional[float]:
    """
    Returns adjusted score or None if hard-excluded.

    Final score = base_score
               × (1 + interest_contribution)
               × (1 + topic_delta)
               × (1 + tone_delta)
               × (1 + sentiment_boost)
               × presentation_multiplier
    """
    if not passes_hard_filters(post, prefs, defaults, viewer_id):
        return None

    t_delta, blocked = topic_score(post, prefs.topics)
    if blocked:
        return None

    apply_interest = defaults.apply_interest_filter
    i_score = interest_score(post, prefs.signals, history) if apply_interest else 0.0
    threshold, proof_boost = social_proof_adjustment(post, prefs)
    interest_contrib = i_score * threshold + proof_boost

    t_score = tone_score(post, prefs.tone)

    sentiment_boost = 0.15 if (
        prefs.included_sentiments and post.sentiment in prefs.included_sentiments
    ) else 0.0

    pres = presentation_multiplier(post, prefs, defaults)

    final = (
        post.base_score
        * (1.0 + interest_contrib)
        * (1.0 + t_delta)
        * (1.0 + t_score)
        * (1.0 + sentiment_boost)
        * pres
    )
    return max(final, 0.0)


# ── Preference store ───────────────────────────────────────────────────────────

class PreferenceStore:
    """
    Persists user filter preferences with pre-computed embeddings.
    Embedding calls happen here (save time), never on the feed hot path.
    """

    def __init__(self, embed_fn):
        # embed_fn(texts: list[str]) -> np.ndarray [n, D]
        self._embed = embed_fn
        self._store: dict[int, UserFilterPrefs] = {}

    def save(
        self,
        user_id:              int,
        prefs:                UserFilterPrefs,
        preferred_topics:     list[str] = (),
        blocked_topics:       list[str] = (),
        raw_tone_description: Optional[str] = None,
    ) -> None:
        """Pre-compute embeddings and store. Called at settings-save time."""
        if preferred_topics:
            vecs = self._embed(list(preferred_topics))
            prefs.topics.preferred_embeddings = [vecs[i] for i in range(len(vecs))]
        if blocked_topics:
            vecs = self._embed(list(blocked_topics))
            prefs.topics.blocked_embeddings = [vecs[i] for i in range(len(vecs))]
        if raw_tone_description:
            vec = self._embed([raw_tone_description])[0]
            prefs.tone.tone_query_embedding = vec / np.linalg.norm(vec)
        self._store[user_id] = prefs

    def load(self, user_id: int) -> Optional[UserFilterPrefs]:
        return self._store.get(user_id)


def _unix_now() -> int:
    return int(time.time())


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1)

    def stub_embed(texts):
        v = rng.standard_normal((len(texts), EMBEDDING_DIM)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    # User prefs: wants analytical posts in English or German, no vulgar style,
    # suppress outraged tone, small accounts preferred, fact-claims only
    prefs = UserFilterPrefs(
        language=LanguagePrefs(
            included_original_languages=["en", "de"],
            show_auto_translated=False,
            show_manually_translated=True,
        ),
        style=StylePrefs(excluded_styles=[ContentStyleGroup.VULGAR]),
        tone=TonePrefs(
            preferred_tones=[ContentTone.ANALYTICAL],
            suppressed_tones=[ContentTone.OUTRAGED],
        ),
        account_size=AccountSizePrefs(included_tiers=[AccountSizeTier.SMALL,
                                                       AccountSizeTier.MEDIUM]),
        claims=ClaimPrefs(included_claim_types=[ClaimType.FACT]),
        included_post_types=[PostType.ORIGINAL, PostType.QUOTE],
    )
    store = PreferenceStore(stub_embed)
    store.save(1, prefs,
               preferred_topics=["AI safety"],
               blocked_topics=["cryptocurrency"],
               raw_tone_description="analytical but not dry")
    loaded = store.load(1)

    history = [
        UserInteraction(EngagementSignal.REPOST, _unix_now() - 7200, ["AI", "AI safety"]),
        UserInteraction(EngagementSignal.LIKE,   _unix_now() - 86400, ["machine learning"]),
    ]

    defaults = FEED_DEFAULTS[FeedContext.FOR_YOU]

    posts = [
        PostFeatures(
            post_id=1, base_score=0.7, created_at_unix=_unix_now()-3600,
            post_type=PostType.ORIGINAL, is_illicit=False, illicit_targets_user_id=None,
            style_group=ContentStyleGroup.INFORMAL,
            tones=[ContentTone.ANALYTICAL], sentiment=Sentiment.NEUTRAL,
            claim_type=ClaimType.FACT, verification_statuses=[VerificationStatus.AI_CHECKED],
            original_language="en", available_translation_targets=[],
            has_manual_translation=False, has_auto_translation=False,
            post_embedding=stub_embed(["AI safety alignment"])[0],
            tone_embedding=TONE_EMBEDDINGS[ContentTone.ANALYTICAL],
            seen_by_n_followed=4, impression_count=600, author_followers=8000,
            topic_ids=["AI safety", "alignment"],
        ),
        PostFeatures(
            post_id=2, base_score=0.9, created_at_unix=_unix_now()-3600,
            post_type=PostType.REPOST, is_illicit=False, illicit_targets_user_id=None,
            style_group=ContentStyleGroup.VULGAR,          # excluded by style
            tones=[ContentTone.OUTRAGED], sentiment=Sentiment.NEGATIVE,
            claim_type=ClaimType.OPINION,
            verification_statuses=[VerificationStatus.NONE],
            original_language="en", available_translation_targets=[],
            has_manual_translation=False, has_auto_translation=False,
            post_embedding=stub_embed(["crypto moon"])[0],
            tone_embedding=TONE_EMBEDDINGS[ContentTone.OUTRAGED],
            seen_by_n_followed=0, impression_count=100000, author_followers=2000000,
            topic_ids=["cryptocurrency"],
        ),
        PostFeatures(
            post_id=3, base_score=0.6, created_at_unix=_unix_now()-3600,
            post_type=PostType.ORIGINAL, is_illicit=False, illicit_targets_user_id=None,
            style_group=ContentStyleGroup.FORMAL,
            tones=[ContentTone.ANALYTICAL, ContentTone.DIPLOMATIC],
            sentiment=Sentiment.NEUTRAL, claim_type=ClaimType.FACT,
            verification_statuses=[VerificationStatus.COMMUNITY],
            original_language="de", available_translation_targets=["en"],
            has_manual_translation=True, has_auto_translation=False,
            post_embedding=stub_embed(["AI regulation policy Germany"])[0],
            tone_embedding=TONE_EMBEDDINGS[ContentTone.ANALYTICAL],
            seen_by_n_followed=2, impression_count=1200, author_followers=5000,
            topic_ids=["AI", "regulation"],
        ),
        PostFeatures(
            post_id=4, base_score=0.5, created_at_unix=_unix_now()-3600,
            post_type=PostType.ORIGINAL, is_illicit=False, illicit_targets_user_id=None,
            style_group=ContentStyleGroup.INFORMAL,
            tones=[ContentTone.NEUTRAL], sentiment=Sentiment.NEUTRAL,
            claim_type=ClaimType.FACT, verification_statuses=[VerificationStatus.NONE],
            original_language="fr", available_translation_targets=["en"],
            has_manual_translation=False, has_auto_translation=True,   # auto-translated
            post_embedding=stub_embed(["French tech news"])[0],
            tone_embedding=TONE_EMBEDDINGS[ContentTone.NEUTRAL],
            seen_by_n_followed=0, impression_count=800, author_followers=3000,
            topic_ids=["tech"],
        ),
    ]

    print(f"Feed: FOR_YOU  |  Viewer: user_1\n{'─'*55}")
    for p in posts:
        s = score_candidate(p, loaded, defaults, history, viewer_id=1)
        reason = ""
        if s is None:
            if p.style_group in loaded.style.excluded_styles:
                reason = f"excluded style={p.style_group.name}"
            elif loaded.included_post_types and p.post_type not in loaded.included_post_types:
                reason = f"post type={p.post_type.name} not in whitelist"
            elif loaded.claims.included_claim_types and p.claim_type not in loaded.claims.included_claim_types:
                reason = f"claim_type={p.claim_type.name}"
            elif p.original_language not in (loaded.language.included_original_languages or []):
                reason = f"language={p.original_language}, auto_translated={p.has_auto_translation}"
            else:
                reason = "hard filtered"
            print(f"  Post {p.post_id}  EXCLUDED  ({reason})")
        else:
            print(f"  Post {p.post_id}  score={s:.3f}  "
                  f"(base={p.base_score}, lang={p.original_language}, "
                  f"tones={[t.name for t in p.tones]})")