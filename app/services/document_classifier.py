"""
Document Classifier Service for automatic document metadata detection.
Analyzes document content and filename to determine technology and domain.
Used during PDF ingestion to tag documents for intelligent query routing.
"""

import json
import re
from typing import Tuple, Optional, Dict, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Resume/CV detection: strong signals before tech keywords (resumes list GPT/RAG/LLM as skills).
RESUME_FILENAME_TOKENS = frozenset(
    {"resume", "cv", "vitae", "curriculum", "biodata"}
)

# Short LLM-related keywords scored with word boundaries only (avoids substring hits).
LLM_BOUNDARY_KEYWORDS = frozenset({"rag", "gpt", "llm"})


class DocumentClassifier:
    """
    Classifies documents to determine their technology and domain.
    Uses content analysis and filename patterns for classification.
    """
    
    # Filename patterns for quick detection
    FILENAME_PATTERNS: Dict[str, List[str]] = {
        "react": ["react", "hooks", "jsx", "nextjs", "next.js", "gatsby"],
        "angular": ["angular", "ng-"],
        "vue": ["vue", "vuejs", "nuxt"],
        "javascript": ["javascript", "js-", "ecmascript", "es6"],
        "typescript": ["typescript", "ts-"],
        "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
        "java": ["java", "spring", "maven", "gradle", "hibernate"],
        "nodejs": ["node", "express", "npm"],
        "ml": ["machine-learning", "ml-", "sklearn", "scikit"],
        "deep_learning": ["deep-learning", "neural", "cnn", "rnn", "lstm"],
        "tensorflow": ["tensorflow", "keras"],
        "pytorch": ["pytorch", "torch"],
        "nlp": ["nlp", "natural-language", "text-processing"],
        "ai": ["artificial-intelligence", "ai-", "gen-ai", "generative"],
        "llm": ["llm", "gpt", "langchain", "rag"],
        "sql": ["sql", "mysql", "postgresql", "database"],
        "docker": ["docker", "kubernetes", "k8s", "container"],
        "aws": ["aws", "amazon", "cloud"],
        "git": ["git", "github", "version-control"],
    }
    
    # Content keywords with weights for detection
    CONTENT_KEYWORDS: Dict[str, Dict[str, int]] = {
        "react": {
            "usestate": 10, "useeffect": 10, "usememo": 8, "usecallback": 8,
            "useref": 8, "usecontext": 8, "usereducer": 8, "jsx": 8,
            "component": 5, "props": 5, "react": 8, "hooks": 7,
            "virtual dom": 6, "redux": 6, "context api": 6
        },
        "angular": {
            "@component": 10, "@injectable": 10, "ngmodule": 10,
            "angular": 8, "rxjs": 7, "observable": 5, "directive": 6
        },
        "vue": {
            "v-model": 10, "v-if": 8, "v-for": 8, "vue": 8,
            "computed property": 6, "vuex": 7, "pinia": 7
        },
        "javascript": {
            "const ": 3, "let ": 3, "var ": 2, "function": 2,
            "arrow function": 5, "promise": 5, "async": 4, "await": 4,
            "closure": 6, "prototype": 5, "callback": 4
        },
        "typescript": {
            "interface": 5, ": string": 6, ": number": 6, ": boolean": 6,
            "type ": 5, "generic": 5, "<t>": 4, "typescript": 8
        },
        "python": {
            "def ": 3, "class ": 3, "import ": 2, "from ": 2,
            "self.": 4, "__init__": 6, "python": 8, "pip": 5,
            "list comprehension": 6, "decorator": 5, "@": 2,
            "django": 8, "flask": 8, "fastapi": 8,
            "pandas": 8, "numpy": 8, "matplotlib": 6, "jupyter": 7,
            "elif ": 4, "except": 4, "async def": 7, "type hints": 6,
            "virtualenv": 6, "pytest": 6,
        },
        "java": {
            "public class": 8, "private": 4, "protected": 4,
            "void": 4, "static": 3, "java": 8, "jvm": 6,
            "spring": 8, "hibernate": 7, "@autowired": 8
        },
        "nodejs": {
            "require(": 6, "module.exports": 8, "express": 8,
            "app.get": 6, "app.post": 6, "middleware": 5,
            "npm": 6, "package.json": 7, "node": 5
        },
        "ml": {
            "machine learning": 10, "supervised": 7, "unsupervised": 7,
            "classification": 6, "regression": 6, "clustering": 6,
            "decision tree": 7, "random forest": 7, "svm": 6,
            "gradient descent": 7, "cross validation": 6, "overfitting": 6,
            "feature": 4, "model": 3, "training": 4, "prediction": 5
        },
        "deep_learning": {
            "neural network": 10, "deep learning": 10, "cnn": 8, "rnn": 8,
            "lstm": 8, "transformer": 8, "attention": 6,
            "backpropagation": 7, "activation function": 6,
            "dropout": 5, "batch normalization": 6, "layer": 4
        },
        "tensorflow": {
            "tensorflow": 10, "tf.": 8, "keras": 8, "tf.keras": 10,
            "sequential": 5, "model.fit": 6
        },
        "pytorch": {
            "pytorch": 10, "torch": 8, "torch.nn": 10,
            "tensor": 5, "autograd": 7, "nn.module": 8
        },
        "nlp": {
            "natural language": 10, "nlp": 10, "tokenization": 8,
            "stemming": 7, "lemmatization": 7, "word embedding": 8,
            "word2vec": 8, "bert": 8, "transformer": 6,
            "sentiment": 6, "named entity": 7
        },
        "ai": {
            "artificial intelligence": 10, "ai": 5, "intelligent": 4,
            "generative ai": 10, "gen ai": 8
        },
        "llm": {
            "large language model": 10, "llm": 10, "gpt": 8,
            "prompt engineering": 9, "rag": 8, "retrieval augmented": 9,
            "fine-tuning": 7, "langchain": 8, "embedding": 5,
            "vector database": 7, "chatgpt": 8
        },
        "sql": {
            "select": 4, "from": 2, "where": 3, "join": 5,
            "insert": 4, "update": 4, "delete": 4, "sql": 8,
            "database": 5, "table": 3, "index": 4, "query": 3
        },
        "docker": {
            "docker": 10, "container": 7, "dockerfile": 10,
            "kubernetes": 9, "k8s": 8, "pod": 6, "deployment": 5
        },
        "aws": {
            "aws": 10, "amazon web services": 10, "ec2": 8, "s3": 8,
            "lambda": 6, "cloudformation": 7, "api gateway": 6
        },
        "git": {
            "git": 8, "github": 7, "commit": 5, "branch": 5,
            "merge": 5, "pull request": 6, "clone": 4, "push": 4
        },
    }
    
    # Domain mapping
    TECHNOLOGY_TO_DOMAIN: Dict[str, str] = {
        "react": "frontend",
        "angular": "frontend",
        "vue": "frontend",
        "javascript": "frontend",
        "typescript": "frontend",
        "html": "frontend",
        "css": "frontend",
        "python": "backend",
        "java": "backend",
        "nodejs": "backend",
        "go": "backend",
        "rust": "backend",
        "csharp": "backend",
        "php": "backend",
        "ruby": "backend",
        "ml": "data-science",
        "deep_learning": "data-science",
        "tensorflow": "data-science",
        "pytorch": "data-science",
        "nlp": "data-science",
        "data_science": "data-science",
        "ai": "ai",
        "llm": "ai",
        "sql": "database",
        "nosql": "database",
        "docker": "devops",
        "aws": "devops",
        "azure": "devops",
        "gcp": "devops",
        "git": "devops",
        "linux": "devops",
    }
    
    def __init__(self):
        """Initialize the document classifier."""
        logger.info("DocumentClassifier initialized")

    @staticmethod
    def _filename_has_token(filename_lower: str, pattern: str) -> bool:
        """
        True if `pattern` appears as a whole token (split on non-alphanumeric), not as a
        substring of a longer word (e.g. react in reactive, node in inode).
        """
        pl = pattern.lower()
        if not pl:
            return False
        if any(c in pl for c in "./\\"):
            return pl in filename_lower
        if "-" in pl and len(pl) > 2:
            return pl in filename_lower
        norm = re.sub(r"[^a-z0-9]+", " ", filename_lower)
        norm = f" {norm.strip()} "
        return f" {pl} " in norm

    def _resume_likelihood(self, filename: str, text: str) -> int:
        """
        Heuristic score for CV/resume PDFs. High scores → classify as general, not course tech.
        """
        score = 0
        if filename:
            base = filename.rsplit(".", 1)[0].lower() if "." in filename else filename.lower()
            norm = re.sub(r"[^a-z0-9]+", " ", base).strip()
            for tok in norm.split():
                if tok in RESUME_FILENAME_TOKENS:
                    score += 14
                    break
                if tok.startswith("resume") and len(tok) <= 12:
                    score += 14
                    break

        if not text or len(text.strip()) < 80:
            return score

        blob = text[:28000]
        low = blob.lower()

        markers = [
            (r"\bwork experience\b", 5),
            (r"\bprofessional experience\b", 5),
            (r"\bemployment history\b", 5),
            (r"\bemployment\b", 2),
            (r"\beducation\b", 3),
            (r"\bacademic background\b", 4),
            (r"\btechnical skills\b", 4),
            (r"\bcore competencies\b", 4),
            (r"\byears of experience\b", 5),
            (r"\bcurriculum vitae\b", 6),
            (r"\bcareer objective\b", 4),
            (r"\bprofessional summary\b", 4),
            (r"\bsummary\b", 1),
            (r"\breferences\b", 2),
            (r"linkedin\.com/in/", 4),
        ]
        for pat, w in markers:
            if re.search(pat, low):
                score += w

        if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", low):
            score += 3
        if re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", blob):
            score += 2

        return score
    
    def classify_from_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Quick classification based on filename patterns.
        
        Args:
            filename: The document filename
            
        Returns:
            Tuple of (technology, domain) or (None, None) if not detected
        """
        if not filename:
            return None, None
        
        filename_lower = filename.lower()
        
        for tech, patterns in self.FILENAME_PATTERNS.items():
            for pattern in patterns:
                matched = (
                    pattern in filename_lower
                    if (any(c in pattern for c in ".\\/") or "-" in pattern)
                    else self._filename_has_token(filename_lower, pattern)
                )
                if matched:
                    domain = self.TECHNOLOGY_TO_DOMAIN.get(tech, "general")
                    logger.info(f"Filename classification: {filename} -> {tech}/{domain}")
                    return tech, domain
        
        return None, None
    
    def classify_from_content(
        self, 
        text: str, 
        sample_size: int = 10000,
        min_score: int = 10,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Classification based on document content analysis.
        
        Args:
            text: Document text content
            sample_size: Maximum characters to analyze (for performance)
            
        Returns:
            Tuple of (technology, domain) or (None, None) if not detected
        """
        if not text or len(text.strip()) < 100:
            return None, None
        
        # Use a sample for performance (beginning, middle, end)
        if len(text) > sample_size:
            sample = (
                text[:sample_size // 3] + 
                text[len(text)//2 - sample_size//6 : len(text)//2 + sample_size//6] +
                text[-sample_size // 3:]
            )
        else:
            sample = text
        
        sample_lower = sample.lower()
        
        # Score each technology based on keyword matches
        tech_scores: Dict[str, int] = Counter()
        
        for tech, keywords in self.CONTENT_KEYWORDS.items():
            for keyword, weight in keywords.items():
                kw = keyword.lower()
                if tech == "llm" and kw in LLM_BOUNDARY_KEYWORDS:
                    count = len(re.findall(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", sample_lower))
                else:
                    count = sample_lower.count(kw)
                if count > 0:
                    score = weight * min(count, 10)
                    tech_scores[tech] += score
        
        # No technology detected
        if not tech_scores:
            return None, None
        
        # Get technology with highest score
        best_tech = tech_scores.most_common(1)[0][0]
        best_score = tech_scores[best_tech]
        
        # Require minimum confidence (score threshold)
        if best_score < min_score:
            logger.info(
                f"Content classification: low confidence (score={best_score}, min={min_score})"
            )
            return None, None
        
        domain = self.TECHNOLOGY_TO_DOMAIN.get(best_tech, "general")
        
        logger.info(f"Content classification: {best_tech}/{domain} (score={best_score})")
        logger.debug(f"All scores: {dict(tech_scores.most_common(5))}")
        
        return best_tech, domain

    @staticmethod
    def _sample_for_classification(text: str, max_chars: int) -> str:
        if not text or max_chars < 1:
            return ""
        t = text.strip()
        if len(t) <= max_chars:
            return t
        third = max_chars // 3
        return (
            t[:third]
            + t[len(t) // 2 - third // 2 : len(t) // 2 + third // 2]
            + t[-third:]
        )

    def _classify_with_prompt(self, filename: str, text: str) -> Optional[Tuple[str, str]]:
        """
        One LLM call with explicit instructions (ChatGPT-style): infer document kind,
        technology, and domain. Used as primary path when DOCUMENT_CLASSIFY_MODE=prompt,
        or at the end of heuristic mode when DOCUMENT_CLASSIFY_USE_LLM=true.
        """
        try:
            from openai import OpenAI
            from app.config import settings, get_completion_client_config
        except Exception as e:
            logger.warning("Document prompt classify: config import failed: %s", e)
            return None

        max_chars = int(getattr(settings, "DOCUMENT_CLASSIFY_LLM_MAX_CHARS", 6000) or 6000)
        if not text or len(text.strip()) < 40:
            return None

        sample = self._sample_for_classification(text, max_chars)
        if len(sample) < 40:
            return None

        try:
            api_key, model, base_url = get_completion_client_config()
        except ValueError as e:
            logger.warning("Document prompt classify skipped: %s", e)
            return None

        tech_keys = sorted(self.TECHNOLOGY_TO_DOMAIN.keys())
        domain_vals = sorted(set(self.TECHNOLOGY_TO_DOMAIN.values()) | {"general"})
        allowed_tech = set(tech_keys) | {"general"}
        allowed_dom = set(domain_vals)

        kinds = [
            "resume_cv",
            "course_or_tutorial",
            "technical_reference",
            "documentation",
            "research_or_whitepaper",
            "business_or_legal",
            "mixed_or_unclear",
        ]

        system = """You label PDFs for a learning/search library (technology + domain).

Reply with ONE JSON object only (no markdown code fences). Keys EXACTLY:
- "document_kind": string, one of the allowed kinds listed by the user.
- "technology": string, must be one of the allowed technologies OR "general".
- "domain": string, must be one of the allowed domains OR "general".

Guidelines:
- **resume_cv**: résumés, CVs, profiles listing jobs/education/skills (even if skills mention GPT, RAG, LangChain). MUST use "general" for BOTH technology and domain.
- **course_or_tutorial**: teaching material with lessons/exercises focused mainly on one stack → pick that stack (e.g. Python course → python + backend).
- **technical_reference / documentation**: manuals, API docs, specs → primary stack they document.
- **research_or_whitepaper**: pick closest stack if clearly one subject; otherwise general/general.
- **business_or_legal**: contracts, policies → usually general/general unless purely technical.
- **mixed_or_unclear**: no single clear primary stack → general/general.

Do not let isolated buzzwords override the overall document type."""

        user_msg = (
            f"Filename: {filename or 'unknown'}\n\n"
            f"Allowed document_kind values: {kinds}\n"
            f"Allowed technology values: {sorted(allowed_tech)}\n"
            f"Allowed domain values: {sorted(allowed_dom)}\n\n"
            "Text excerpt (may be partial OCR):\n---\n"
            f"{sample}\n---"
        )

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        client = OpenAI(**kwargs)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=220,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("Document prompt classify API error: %s", e)
            return None

        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                logger.warning("Document prompt classify: bad JSON: %s", raw[:200])
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        kind = str(data.get("document_kind", "") or "").strip().lower()
        tech = str(data.get("technology", "general")).strip().lower()
        dom = str(data.get("domain", "general")).strip().lower()

        if kind in ("resume_cv", "business_or_legal", "mixed_or_unclear"):
            tech, dom = "general", "general"
        else:
            if tech not in allowed_tech:
                tech = "general"
            if dom not in allowed_dom:
                dom = self.TECHNOLOGY_TO_DOMAIN.get(tech, "general")

        logger.info(
            "Prompt document classification: %s -> kind=%s %s/%s",
            filename,
            kind or "?",
            tech,
            dom,
        )
        return tech, dom

    def _run_heuristic_classify(
        self,
        filename: str,
        text: str,
        *,
        min_score: int,
        use_llm_fallback: bool,
        resume_threshold: int,
    ) -> Tuple[str, str]:
        """Resume heuristics, filename tokens, keyword scores, optional prompt at end."""
        if self._resume_likelihood(filename, text) >= resume_threshold:
            logger.info(
                "Document looks like a resume/CV -> general/general (threshold=%s)",
                resume_threshold,
            )
            return "general", "general"

        tech, domain = self.classify_from_filename(filename)
        if tech:
            return tech, domain

        tech, domain = self.classify_from_content(text, min_score=min_score)
        if tech:
            return tech, domain

        if use_llm_fallback:
            llm_result = self._classify_with_prompt(filename, text)
            if llm_result:
                return llm_result

        logger.info(f"Document classification: defaulting to general for {filename}")
        return "general", "general"

    def classify(
        self,
        filename: str,
        text: str,
        min_content_score: Optional[int] = None,
        use_llm_fallback: Optional[bool] = None,
        classify_mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Classify document for technology/domain tags.

        DOCUMENT_CLASSIFY_MODE (or classify_mode override):
        - **prompt**: one LLM classification first (natural-language instructions), then heuristics if it fails.
        - **heuristic**: keyword/filename/resume heuristics; optional LLM only if DOCUMENT_CLASSIFY_USE_LLM.

        Args:
            classify_mode: Override env DOCUMENT_CLASSIFY_MODE for tests ("prompt" | "heuristic").
        """
        from app.config import settings

        min_score = (
            min_content_score
            if min_content_score is not None
            else int(getattr(settings, "DOCUMENT_CLASSIFY_MIN_SCORE", 10) or 10)
        )
        do_llm = (
            use_llm_fallback
            if use_llm_fallback is not None
            else bool(getattr(settings, "DOCUMENT_CLASSIFY_USE_LLM", False))
        )

        resume_threshold = int(getattr(settings, "DOCUMENT_CLASSIFY_RESUME_THRESHOLD", 9) or 9)

        mode = (classify_mode or getattr(settings, "DOCUMENT_CLASSIFY_MODE", "heuristic") or "heuristic").lower().strip()
        if mode not in ("prompt", "heuristic"):
            logger.warning("Unknown DOCUMENT_CLASSIFY_MODE=%r; using heuristic", mode)
            mode = "heuristic"

        if mode == "prompt":
            prompted = self._classify_with_prompt(filename, text)
            if prompted:
                return prompted
            logger.info("Prompt classification failed or unavailable; falling back to heuristics")

        return self._run_heuristic_classify(
            filename,
            text,
            min_score=min_score,
            use_llm_fallback=do_llm,
            resume_threshold=resume_threshold,
        )


# Global singleton instance
_classifier: Optional[DocumentClassifier] = None


def get_document_classifier() -> DocumentClassifier:
    """Get or create the global document classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier


def classify_document(filename: str, text: str, **kwargs) -> Tuple[str, str]:
    """
    Convenience function to classify a document.
    Pass-through kwargs match DocumentClassifier.classify (e.g. classify_mode=).
    """
    classifier = get_document_classifier()
    return classifier.classify(filename, text, **kwargs)

