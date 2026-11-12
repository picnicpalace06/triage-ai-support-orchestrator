"""TF-IDF based retriever over corpus passages.

Deliberately dependency-light: scikit-learn's ``TfidfVectorizer`` + cosine
similarity.  This works fully offline and is fast for a corpus of a few
thousand passages, which is the regime the support sites fall into.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .corpus import Corpus, Passage

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    passage: Passage
    score: float


class TfidfRetriever:
    def __init__(self, corpus: Corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer  # local import

        self.corpus = corpus
        self.passages: list[Passage] = corpus.passages
        if not self.passages:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )
        texts = [_normalize(p.text) for p in self.passages]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        *,
        company: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if not self.passages or self._vectorizer is None:
            return []
        q = _expand_synonyms(_normalize(query))
        if not q.strip():
            return []
        qv = self._vectorizer.transform([q])
        # cosine because tf-idf rows are L2 normalised by default
        sims = (self._matrix @ qv.T).toarray().ravel()

        # Apply company filter
        if company and company.lower() != "none":
            mask = np.array(
                [p.company == company.lower() for p in self.passages], dtype=bool
            )
            sims = np.where(mask, sims, -1.0)

        # Boost passages whose title shares words with the query.
        q_words = set(re.findall(r"\w+", q.lower()))
        if q_words:
            for i, p in enumerate(self.passages):
                t_words = set(re.findall(r"\w+", p.doc_title.lower()))
                overlap = len(q_words & t_words)
                if overlap:
                    sims[i] += 0.05 * overlap

        order = np.argsort(-sims)
        hits: list[RetrievalHit] = []
        for idx in order[: top_k * 3]:  # gather extra to dedup by document
            score = float(sims[idx])
            if score <= 0:
                continue
            hits.append(RetrievalHit(self.passages[idx], score))
        # Deduplicate so that the top-k cover several documents.
        seen: set[str] = set()
        deduped: list[RetrievalHit] = []
        for h in hits:
            key = h.passage.doc_path
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
            if len(deduped) >= top_k:
                break
        return deduped


def _normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Simple bidirectional synonym map to close common phrasing gaps between
# user tickets and corpus phrasing (e.g. "employee" vs "team member").
# Each (pattern, expansion) applies only to the query, never to documents.
_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(employees?|teammates?|colleagues?|coworkers?|staff)\b", re.I),
     "team member user recruiter interviewer"),
    (re.compile(r"\b(hiring account|company account|work account|org(?:anisation|anization)? account)\b", re.I),
     "team company HackerRank for Work account"),
    (re.compile(r"\b(left the company|has left|resigned|quit|departed|terminated)\b", re.I),
     "remove deactivate lock revoke access"),
    (re.compile(r"\b(remove|delete|disable|deactivate|revoke|kick out)\b", re.I),
     "remove delete deactivate lock"),
    (re.compile(r"\b(seat|license|licence)\b", re.I),
     "user role entitlement seat license"),
    (re.compile(r"\b(reschedule|rescheduling|postpone|new (?:test )?date)\b", re.I),
     "reschedule extend invite candidate test"),
    (re.compile(r"\b(stuck|frozen|hangs?|hung|not (?:working|loading|responding))\b", re.I),
     "issue troubleshoot error not working"),
    # Map "remove an interviewer / user from the platform" to user-management docs
    (re.compile(r"\bremove\s+(?:an?\s+)?(?:interviewer|user|recruiter|admin)\b", re.I),
     "lock user access deactivate teams management remove team member"),
    (re.compile(r"\bthree\s+dots\b|\bellipsis\b|\baction\s+menu\b", re.I),
     "actions teams management more options"),
    # Certificate name update -> certifications FAQ doc
    (re.compile(r"\b(certificate|certification)s?\b.{0,30}\b(name|wrong|incorrect|update|change|spelling)\b", re.I),
     "certifications faq update name certificate"),
    (re.compile(r"\b(name)\b.{0,30}\b(certificate|certification)s?\b", re.I),
     "certifications faq update name certificate"),
    # Data retention questions for Claude
    (re.compile(r"\b(how long|retention|retain|store(?:d)?|kept)\b.{0,40}\b(data|conversation|chat)\b", re.I),
     "data retention privacy policy how long store"),
]


def _expand_synonyms(query: str) -> str:
    extras: list[str] = []
    for pattern, expansion in _SYNONYMS:
        if pattern.search(query):
            extras.append(expansion)
    if extras:
        return query + " " + " ".join(extras)
    return query


def best_score(hits: Sequence[RetrievalHit]) -> float:
    return hits[0].score if hits else 0.0
