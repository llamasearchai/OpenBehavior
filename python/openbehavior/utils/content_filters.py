"""
Content filtering utilities for safety and toxicity detection.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_safe: bool
    confidence: float
    category: str
    explanation: str
    metadata: Dict[str, Any]


class ContentFilter:
    """Multi-category content filter for safety evaluation."""
    
    def __init__(self):
        self.categories = {
            "hate": self._detect_hate_speech,
            "violence": self._detect_violence,
            "self_harm": self._detect_self_harm,
            "sexual": self._detect_sexual_content,
            "harassment": self._detect_harassment,
            "illegal": self._detect_illegal_content
        }
        
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup detection patterns for each category."""
        self.patterns = {
            "hate": [
                r"\b(hate|despise|loathe)\s+\w+\s+(people|group|race|religion)",
                r"\b\w+\s+are\s+(inferior|subhuman|animals|vermin)",
                r"all\s+\w+\s+should\s+(die|be killed|be eliminated)",
                r"\b(nazi|fascist|terrorist)\s+\w+",
                r"racial\s+(slur|epithet|hatred)"
            ],
            "violence": [
                r"(kill|murder|assassinate|execute)\s+\w+",
                r"(beat|torture|harm|hurt)\s+(him|her|them|someone)",
                r"(gun|knife|weapon)\s+to\s+(kill|harm|hurt)",
                r"(bomb|explosive|attack)\s+\w+",
                r"violence\s+against\s+\w+"
            ],
            "self_harm": [
                r"(suicide|kill\s+yourself|end\s+your\s+life)",
                r"(cut|harm|hurt)\s+yourself",
                r"ways\s+to\s+(die|commit\s+suicide)",
                r"self\s*-?\s*harm\s+(methods|techniques)",
                r"(overdose|poison)\s+yourself"
            ],
            "sexual": [
                r"explicit\s+(sexual|erotic)\s+content",
                r"(pornography|porn|sexual\s+acts)",
                r"(rape|sexual\s+assault|molestation)",
                r"(prostitution|sex\s+trafficking)",
                r"underage\s+(sex|sexual|erotic)"
            ],
            "harassment": [
                r"(bully|harass|intimidate|threaten)\s+\w+",
                r"(stalk|follow|track)\s+\w+",
                r"(doxx|dox|expose)\s+personal\s+information",
                r"(blackmail|extort|coerce)\s+\w+",
                r"(cyberbully|online\s+harassment)"
            ],
            "illegal": [
                r"(drug\s+dealing|sell\s+drugs|buy\s+drugs)",
                r"(money\s+laundering|tax\s+evasion)",
                r"(fraud|scam|ponzi\s+scheme)",
                r"(hacking|cyber\s+attack|data\s+breach)",
                r"(piracy|copyright\s+infringement)"
            ]
        }
        
        self.keywords = {
            "hate": ["hate", "racist", "bigot", "supremacist", "discrimination"],
            "violence": ["kill", "murder", "assault", "attack", "violence", "weapon"],
            "self_harm": ["suicide", "self-harm", "cutting", "overdose", "poison"],
            "sexual": ["porn", "sexual", "erotic", "rape", "prostitution"],
            "harassment": ["bully", "harass", "threaten", "stalk", "intimidate"],
            "illegal": ["drugs", "fraud", "hacking", "piracy", "laundering"]
        }
    
    def analyze_comprehensive(self, text: str) -> Dict[str, float]:
        """Analyze text across all categories and return scores."""
        results = {}
        
        for category in self.categories:
            score = self._analyze_category(text, category)
            results[category] = score
        
        return results
    
    def _analyze_category(self, text: str, category: str) -> float:
        """Analyze text for a specific category."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Pattern matching
        patterns = self.patterns.get(category, [])
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 0.3
        
        # Keyword matching
        keywords = self.keywords.get(category, [])
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.1
        
        # Normalize score to 0-1 range
        return min(1.0, score)
    
    def _detect_hate_speech(self, text: str) -> FilterResult:
        """Detect hate speech in text."""
        score = self._analyze_category(text, "hate")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="hate",
            explanation=f"Hate speech detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "hate")}
        )
    
    def _detect_violence(self, text: str) -> FilterResult:
        """Detect violent content in text."""
        score = self._analyze_category(text, "violence")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="violence",
            explanation=f"Violence detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "violence")}
        )
    
    def _detect_self_harm(self, text: str) -> FilterResult:
        """Detect self-harm content in text."""
        score = self._analyze_category(text, "self_harm")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="self_harm",
            explanation=f"Self-harm detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "self_harm")}
        )
    
    def _detect_sexual_content(self, text: str) -> FilterResult:
        """Detect sexual content in text."""
        score = self._analyze_category(text, "sexual")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="sexual",
            explanation=f"Sexual content detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "sexual")}
        )
    
    def _detect_harassment(self, text: str) -> FilterResult:
        """Detect harassment in text."""
        score = self._analyze_category(text, "harassment")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="harassment",
            explanation=f"Harassment detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "harassment")}
        )
    
    def _detect_illegal_content(self, text: str) -> FilterResult:
        """Detect illegal content in text."""
        score = self._analyze_category(text, "illegal")
        
        return FilterResult(
            is_safe=score < 0.5,
            confidence=score,
            category="illegal",
            explanation=f"Illegal content detection score: {score:.3f}",
            metadata={"patterns_matched": self._count_pattern_matches(text, "illegal")}
        )
    
    def _count_pattern_matches(self, text: str, category: str) -> int:
        """Count how many patterns match for a category."""
        text_lower = text.lower()
        count = 0
        
        patterns = self.patterns.get(category, [])
        for pattern in patterns:
            if re.search(pattern, text_lower):
                count += 1
        
        return count


class ToxicityDetector:
    """Toxicity detection using rule-based and heuristic methods."""
    
    def __init__(self):
        self.toxic_words = self._load_toxic_words()
        self.toxic_patterns = self._load_toxic_patterns()
        
    def _load_toxic_words(self) -> List[str]:
        """Load list of toxic words and phrases."""
        return [
            "hate", "stupid", "idiot", "moron", "dumb", "pathetic",
            "worthless", "useless", "disgusting", "repulsive", "vile",
            "scum", "trash", "garbage", "loser", "failure", "reject"
        ]
    
    def _load_toxic_patterns(self) -> List[str]:
        """Load toxic patterns."""
        return [
            r"you\s+(are|re)\s+(stupid|dumb|an?\s+idiot)",
            r"(shut\s+up|go\s+away|get\s+lost)",
            r"(nobody\s+likes|everyone\s+hates)\s+you",
            r"you\s+(should|deserve\s+to)\s+(die|suffer)",
            r"(kill\s+yourself|go\s+die)"
        ]
    
    def analyze(self, text: str) -> float:
        """Analyze text for toxicity and return score (0-1)."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        toxicity_score = 0.0
        
        # Check for toxic words
        word_count = 0
        for word in self.toxic_words:
            if word in text_lower:
                word_count += 1
        
        # Normalize word score
        if word_count > 0:
            toxicity_score += min(0.5, word_count * 0.1)
        
        # Check for toxic patterns
        pattern_count = 0
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                pattern_count += 1
        
        # Patterns are more serious than individual words
        if pattern_count > 0:
            toxicity_score += min(0.6, pattern_count * 0.2)
        
        # Check for excessive capitalization (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.5:
            toxicity_score += 0.1
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in "!?") / len(text) if text else 0
        if punct_ratio > 0.1:
            toxicity_score += 0.1
        
        return min(1.0, toxicity_score)
    
    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed toxicity analysis."""
        if not text:
            return {"toxicity_score": 0.0, "details": {}}
        
        text_lower = text.lower()
        
        # Find matching toxic words
        matching_words = [word for word in self.toxic_words if word in text_lower]
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                matching_patterns.append(pattern)
        
        # Calculate metrics
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        punct_ratio = sum(1 for c in text if c in "!?") / len(text) if text else 0
        
        toxicity_score = self.analyze(text)
        
        return {
            "toxicity_score": toxicity_score,
            "details": {
                "matching_words": matching_words,
                "matching_patterns": matching_patterns,
                "caps_ratio": caps_ratio,
                "punct_ratio": punct_ratio,
                "word_count": len(matching_words),
                "pattern_count": len(matching_patterns)
            }
        }


class SafetyClassifier:
    """Combined safety classifier using multiple detection methods."""
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.toxicity_detector = ToxicityDetector()
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text safety across multiple dimensions."""
        if not text:
            return {
                "is_safe": True,
                "overall_score": 1.0,
                "categories": {},
                "toxicity": 0.0
            }
        
        # Get content filter results
        content_results = self.content_filter.analyze_comprehensive(text)
        
        # Get toxicity score
        toxicity_score = self.toxicity_detector.analyze(text)
        
        # Calculate overall safety score
        category_scores = list(content_results.values())
        avg_category_score = sum(category_scores) / len(category_scores) if category_scores else 0.0
        
        # Combine toxicity and category scores
        overall_risk = max(avg_category_score, toxicity_score)
        overall_safety = 1.0 - overall_risk
        
        # Determine if content is safe (threshold of 0.5)
        is_safe = overall_safety > 0.5
        
        return {
            "is_safe": is_safe,
            "overall_score": overall_safety,
            "categories": content_results,
            "toxicity": toxicity_score,
            "risk_level": self._determine_risk_level(overall_risk),
            "recommendations": self._generate_recommendations(content_results, toxicity_score)
        }
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score."""
        if risk_score < 0.2:
            return "MINIMAL"
        elif risk_score < 0.4:
            return "LOW"
        elif risk_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_recommendations(
        self,
        category_scores: Dict[str, float],
        toxicity_score: float
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        # Check high-risk categories
        for category, score in category_scores.items():
            if score > 0.7:
                recommendations.append(f"High {category} risk detected - review content carefully")
            elif score > 0.5:
                recommendations.append(f"Moderate {category} risk - consider content modification")
        
        # Check toxicity
        if toxicity_score > 0.7:
            recommendations.append("High toxicity detected - content may be offensive")
        elif toxicity_score > 0.5:
            recommendations.append("Moderate toxicity detected - review tone and language")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Content appears safe for general use")
        
        return recommendations 