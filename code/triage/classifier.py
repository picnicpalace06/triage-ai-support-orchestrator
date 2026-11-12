"""Heuristic classifier + safety policy.

We use a lightweight rule-based system on top of which an LLM can refine
the decision.  The rules alone produce decent labels and are 100% offline,
so the agent always behaves predictably.

Outputs of :func:`classify`:

- ``request_type``: ``product_issue`` | ``feature_request`` | ``bug`` | ``invalid``
- ``risk``: ``low`` | ``medium`` | ``high``
- ``inferred_company``: one of the known companies, or ``"none"``
- ``out_of_scope``: bool
- ``triggers``: list of human-readable reasons that fired
- ``injection``: bool – prompt-injection / jailbreak attempt detected
- ``language_non_english``: bool

The triage agent uses these signals to pick between ``replied`` and
``escalated`` and to filter what to send to the response generator.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

COMPANY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hackerrank": (
        "hackerrank", "hacker rank", "hacker-rank", "codepair", "code pair",
        "test invite", "assessment", "candidate report", "interview platform",
        "screen", "proctor", "rescheduling assessment", "hackos",
    ),
    "claude": (
        "claude", "anthropic", "claude.ai", "constitutional ai",
        "claude pro", "claude team", "claude workspace", "bedrock", "vertex",
    ),
    "visa": (
        "visa", "visa card", "credit card", "debit card", "atm", "merchant",
        "chargeback", "interchange", "card issuer", "traveller's cheque",
        "traveler's cheque", "travelers cheque", "card stolen",
    ),
}

# Strong "this is a bug / outage" signals.
BUG_PATTERNS = (
    r"\bnot\s*working\b", r"\bnot\s*responding\b", r"\bdown\b",
    r"\bbroken\b", r"\bfail(ing|ed|s)?\b", r"\berror\b",
    r"\bcrash(ing|ed|es)?\b", r"\boutage\b", r"\bcan'?t\s+(?:access|see|open|use|load)\b",
    r"\bunable to\b", r"\bstopped\s+working\b", r"\bblocker\b",
    r"\binaccessible\b", r"\bblank\s+page\b", r"\b502\b", r"\b500\s*error\b",
)

FEATURE_REQUEST_PATTERNS = (
    r"\bplease\s+add\b", r"\bcan you\s+(?:add|build|support)\b",
    r"\bfeature\s+request\b", r"\bit would be (?:nice|great)\b",
    r"\bwish list\b", r"\bsuggestion\b",
)

# High-risk / sensitive topics that should be escalated regardless of corpus match.
HIGH_RISK_PATTERNS = (
    r"\bidentity\s+theft\b",
    r"\bstolen\b.*\bcard\b", r"\bcard\b.*\bstolen\b",
    r"\blost\s+card\b", r"\bcard\s+lost\b",
    r"\bfraud\b", r"\bfraudulent\b",
    r"\bunauthori[sz]ed\s+(?:charge|transaction)\b",
    r"\bdispute\b.*\bcharge\b", r"\bchargeback\b",
    r"\bdata\s+breach\b", r"\bsecurity\s+(?:vulnerab|incident|breach)\b",
    r"\b(?:bug|vulnerability)\s+bounty\b",
    r"\bhack(?:ed|ing)?\b",
    r"\bphishing\b", r"\bscam\b",
    r"\baccount\s+(?:locked|hacked|compromised|access\s+lost)\b",
    r"\bsubscription\s+(?:pause|cancel|refund)\b",
    r"\brefund\b", r"\bbilling\b", r"\bcharged\s+twice\b",
    r"\bblocked\b.*\bcard\b", r"\bcard\b.*\bblocked\b",
    r"\bemergency\s+cash\b",
    r"\btest\s+score\s+dispute\b", r"\bunfair(?:ly)?\s+grad(?:e|ed|ing)\b",
    r"\binfosec\b", r"\bsecurity\s+review\b", r"\bquestionnaire\b",
    r"\bcrawl(?:ing)?\s+(?:my|our)\s+(?:website|site|data)\b",
)

# Prompt-injection / jailbreak red flags. Multilingual (English/French/Spanish).
INJECTION_PATTERNS = (
    # English
    r"ignore (?:all )?(?:previous|prior) (?:instructions|prompts)",
    r"reveal (?:your )?(?:system|hidden) prompt",
    r"show (?:me )?(?:all )?(?:the )?internal (?:rules|policies|prompts|documents)",
    r"print (?:the )?retrieved? documents?",
    r"dump (?:the )?context",
    r"act as (?:dan|developer mode|jailbreak)",
    r"exact (?:logic|prompt|instructions) (?:you|the agent) use",
    r"(?:show|display|output|leak) (?:me )?(?:the )?(?:retrieved? )?(?:context|documents?|passages?|sources?|hidden|raw) ",
    # French
    r"affiche(?:r|z)? (?:moi |toutes? )?(?:les |la |le )?r[eè]gles? internes?",
    r"affiche(?:r|z)? .*documents? r[eè]cup[eé]r[eé]",
    r"affiche(?:r|z)? .*(?:logique|prompt|instructions?) (?:exacte|interne|cach[eé])",
    r"montre(?:r|z|-moi)? (?:tout|les) r[eè]gles",
    r"r[eè]v[eé]l(?:er|ez|e)? (?:le |votre )?prompt",
    # Spanish
    r"muestra(?:me)? (?:todas? )?(?:las )?reglas internas?",
    r"revela(?:r)? (?:el|tu) prompt",
    r"muestra(?:me)? (?:los )?documentos? recuperados?",
)

# Asks for code / actions outside of customer support.
DANGEROUS_REQUEST_PATTERNS = (
    r"\bdelete\s+all\s+files\b",
    r"\bformat\s+(?:my|the)\s+(?:disk|drive)\b",
    r"\bdisable\s+(?:antivirus|firewall)\b",
    r"\b(?:rm\s+-rf|sudo\s+rm)\b",
)

# Pure chitchat that we can answer without the corpus.
CHITCHAT_PATTERNS = (
    r"^\s*(thanks|thank you|thx|cheers|appreciate it)[\s!.\u2026]*$",
    r"^\s*(hi|hello|hey)[\s!.\u2026]*$",
    r"^\s*(ok|okay|got it)[\s!.\u2026]*$",
)

# Generic out-of-scope (e.g. trivia)
OFF_TOPIC_PATTERNS = (
    r"\bwho\s+(?:is|was)\s+the\s+actor\b",
    r"\bwhat\s+is\s+the\s+name\s+of\s+the\s+actor\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\bweather\s+(?:in|today)\b",
    r"\bwrite\s+me\s+a\s+poem\b",
)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class Classification:
    request_type: str = "product_issue"
    risk: str = "low"
    inferred_company: str = "none"
    out_of_scope: bool = False
    chitchat: bool = False
    injection: bool = False
    language_non_english: bool = False
    triggers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    return text.strip()


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _matches(patterns, text: str) -> list[str]:
    out = []
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            out.append(p)
    return out


def _is_mostly_non_english(text: str) -> bool:
    """Very rough heuristic: if more than 25% of letter chars are outside
    the basic Latin/ASCII range and the text is non-trivially long, treat
    as non-English."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) > 0.25:
        return True
    # Detect common non-English function words even when ASCII only
    lower = text.lower()
    foreign_markers = (
        "bonjour", " ma carte ", "tarjeta", "señor", "señora", "guten tag",
        "हैलो", "你好", "こんにちは", " est ", " mon ", " votre ",
    )
    return any(m in lower for m in foreign_markers)


def infer_company(issue: str, subject: str, given: str | None) -> str:
    given_norm = (given or "").strip().lower()
    if given_norm in ("hackerrank", "claude", "visa"):
        return given_norm
    text = f"{subject}\n{issue}".lower()
    scores = {c: 0 for c in COMPANY_KEYWORDS}
    for company, kws in COMPANY_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[company] += 1
    best = max(scores.values())
    if best == 0:
        return "none"
    # Tie-break by listing order; deterministic.
    for c, s in scores.items():
        if s == best:
            return c
    return "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(issue: str, subject: str = "", company: str | None = None) -> Classification:
    issue = _norm(issue)
    subject = _norm(subject)
    blob = f"{subject}\n{issue}".strip()
    triggers: list[str] = []

    cls = Classification()

    # 1. Prompt injection / jailbreak
    if _any(INJECTION_PATTERNS, blob):
        cls.injection = True
        triggers.append("prompt_injection_attempt")

    # 2. Dangerous off-platform requests (e.g. "give me code to delete all files")
    if _any(DANGEROUS_REQUEST_PATTERNS, blob):
        cls.out_of_scope = True
        cls.request_type = "invalid"
        triggers.append("dangerous_off_platform_request")

    # 3. Pure chitchat
    short = blob.lower()
    if _any(CHITCHAT_PATTERNS, short):
        cls.chitchat = True
        cls.request_type = "invalid"
        triggers.append("chitchat")

    # 4. Off-topic trivia
    if _any(OFF_TOPIC_PATTERNS, blob):
        cls.out_of_scope = True
        cls.request_type = "invalid"
        triggers.append("off_topic")

    # 5. Inferred company
    cls.inferred_company = infer_company(issue, subject, company)

    # 6. Non-English (informational signal)
    cls.language_non_english = _is_mostly_non_english(blob)
    if cls.language_non_english:
        triggers.append("non_english_content")

    # 7. Risk assessment
    risk_hits = _matches(HIGH_RISK_PATTERNS, blob)
    if risk_hits:
        cls.risk = "high"
        triggers.append("high_risk_keywords")
    elif _any(BUG_PATTERNS, blob):
        cls.risk = "medium"
        triggers.append("possible_bug_or_outage")

    # 8. Request type (only if not already set by chitchat / off-topic)
    if cls.request_type not in ("invalid",):
        if _any(FEATURE_REQUEST_PATTERNS, blob):
            cls.request_type = "feature_request"
            triggers.append("feature_request_phrasing")
        elif _any(BUG_PATTERNS, blob):
            cls.request_type = "bug"
        else:
            cls.request_type = "product_issue"

    cls.triggers = triggers
    return cls


# Topics that we always escalate, even if the corpus has a related article,
# because they require a human to verify identity / take action.
ALWAYS_ESCALATE_TOPICS = (
    "identity_theft",
    "lost_or_stolen_card_with_action",
    "fraud_or_unauthorized_charge",
    "security_vulnerability_report",
    "billing_dispute",
    "subscription_change",
    "test_score_dispute",
    "infosec_questionnaire",
    "data_crawl_request",
    "outage_report",
)


def should_always_escalate(issue: str, subject: str, classification: Classification) -> tuple[bool, str]:
    """Return (escalate?, topic_label)."""
    blob = f"{subject}\n{issue}".lower()

    if classification.injection:
        return True, "prompt_injection_attempt"

    if re.search(r"\bidentity\s+theft\b", blob):
        return True, "identity_theft"

    if (
        re.search(r"(stolen|lost|blocked|bloqu[eé]e?|robad[ao]|perdid[ao])\b.{0,40}(card|visa|carte|tarjeta)", blob)
        or re.search(r"(card|visa|carte|tarjeta)\b.{0,40}(stolen|lost|blocked|bloqu[eé]e?|robad[ao]|perdid[ao])", blob)
    ):
        # Lost/stolen/blocked card needs a human / issuer call urgently.
        return True, "lost_or_stolen_card"

    if re.search(r"\bfraud(ulent)?\b|\bunauthori[sz]ed\s+(charge|transaction)\b", blob):
        return True, "fraud_or_unauthorized_charge"

    if re.search(r"\bsecurity\s+vulnerab|\bbug\s+bounty\b|\bvulnerability\s+bounty\b", blob):
        return True, "security_vulnerability_report"

    if re.search(r"\brefund\b|\bcharged\s+twice\b|\border\s+id\b|\bpayment\b.*\b(issue|problem|failed)\b", blob):
        return True, "billing_or_payment_dispute"

    if re.search(r"\bsubscription\s+(pause|cancel|change|downgrade|upgrade)\b|\bpause\s+(?:our|my)\s+subscription\b", blob):
        return True, "subscription_change"

    if re.search(r"\b(test|assessment)\s+score\s+dispute\b|\bgrade(?:d)?\s+unfairly\b|\bincrease\s+my\s+score\b", blob):
        return True, "test_score_dispute"

    if re.search(r"\binfosec\b|\bsecurity\s+(review|questionnaire)\b|\bfilling\s+in\s+the\s+forms?\b", blob):
        return True, "infosec_questionnaire"

    if re.search(r"\b(stop|prevent).{0,30}crawl|\bcrawl(ing)?\s+(my|our)\s+(website|site|data)\b", blob):
        return True, "data_crawl_request"

    if re.search(r"\bsite\s+is\s+down\b|\bnone\s+of\s+the\s+(submissions|pages)\b|\boutage\b|\ball\s+requests\s+(?:are\s+)?failing\b|\bstopped\s+working\s+completely\b", blob):
        return True, "outage_report"

    if re.search(r"\baccount\s+(locked|hacked|compromised)\b|\baccess\s+lost\b|\brestore\s+(my|our)\s+access\b|\bremoved?\s+my\s+seat\b", blob):
        return True, "account_access_recovery"

    if re.search(r"\bemergency\s+cash\b|\burgent\s+(?:need\s+)?cash\b", blob):
        return True, "emergency_cash_request"

    if re.search(
        r"\breschedul(?:e|ing)\b.{0,80}\b(assessment|test|interview|exam)\b|\b(assessment|test|interview|exam)\b.{0,80}\breschedul(?:e|ing)\b|\brequest(?:ing)?\s+(?:a|an)\s+reschedul",
        blob,
    ):
        # HackerRank's policy: HR cannot reschedule on the candidate's behalf;
        # candidate must contact the recruiter directly.
        return True, "reschedule_request"

    if re.search(r"\b(remove|delete)\s+(an?\s+)?(interviewer|user|employee|member|seat)\b", blob):
        # This one IS in the HackerRank docs but if a human-action like a
        # left-employee is described and the user couldn't find the option,
        # we still answer (do not auto-escalate).
        return False, ""

    return False, ""
