"""
Query Classifier Service for intelligent query routing.
Detects technology and domain from user queries to enable scoped vector search.
Uses LLM-based classification for accuracy; falls back to keyword-based on failure.
"""

import re
from typing import Tuple, Optional, Dict, List
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies user queries to determine relevant technology and domain.
    Uses LLM-based detection for accurate scope (e.g. hashmap->java, dictionary->python);
    falls back to keyword-based when LLM is unavailable or returns nothing.
    """
    
    # Technology keywords mapping
    TECHNOLOGY_KEYWORDS: Dict[str, List[str]] = {
        # Frontend Technologies
        "react": [
            "react", "reactjs", "react.js", "usestate", "useeffect", "usememo",
            "usecallback", "useref", "usecontext", "usereducer", "hooks", "jsx",
            "component", "props", "state management", "redux", "context api",
            "virtual dom", "react router", "next.js", "nextjs", "gatsby"
        ],
        "angular": [
            "angular", "angularjs", "typescript angular", "ng-", "ngmodule",
            "component angular", "directive", "pipe angular", "rxjs", "ngrx",
            "angular cli", "angular material"
        ],
        "vue": [
            "vue", "vuejs", "vue.js", "vuex", "pinia", "composition api",
            "options api", "vue router", "nuxt", "nuxtjs"
        ],
        "javascript": [
            "javascript", "js", "ecmascript", "es6", "es2015", "promise",
            "async await", "closure", "prototype", "this keyword", "arrow function",
            "dom manipulation", "event loop", "callback"
        ],
        "typescript": [
            "typescript", "ts", "type annotation", "interface typescript",
            "generic typescript", "type inference", "union type", "intersection type"
        ],
        "html": [
            "html", "html5", "semantic html", "dom", "html element", "html tag",
            "form html", "input html"
        ],
        "css": [
            "css", "css3", "flexbox", "grid css", "media query", "responsive design",
            "sass", "scss", "less", "tailwind", "bootstrap", "styled-components"
        ],
        
        # Backend Technologies
        "python": [
            "python", "python3", "django", "flask", "fastapi", "pandas", "numpy",
            "scipy", "matplotlib", "seaborn", "pytest", "pip", "virtualenv",
            "list comprehension", "decorator python", "decorators", "decorator",
            "generator python", "asyncio", "pydantic", "sqlalchemy",
            "dictionary", "dict python", "python dict", "python dictionary"
        ],
        "java": [
            "java", "jvm", "spring", "spring boot", "maven", "gradle", "hibernate",
            "jpa", "servlet", "jsp", "jdbc", "java stream", "lambda java",
            "collection java", "collection framework", "collections framework",
            "multithreading java",
            "hashmap", "hash map", "hashmap in java", "hash table", "hashtable",
            "hashmap java", "map java"
        ],
        "nodejs": [
            "node", "nodejs", "node.js", "express", "expressjs", "npm", "yarn",
            "package.json", "middleware node", "event-driven"
        ],
        "go": [
            "golang", "go language", "goroutine", "channel go", "go module"
        ],
        "rust": [
            "rust", "cargo rust", "ownership rust", "borrowing rust"
        ],
        "csharp": [
            "c#", "csharp", ".net", "dotnet", "asp.net", "entity framework", "linq"
        ],
        "php": [
            "php", "laravel", "symfony", "composer php"
        ],
        "ruby": [
            "ruby", "rails", "ruby on rails", "gem ruby"
        ],
        
        # Data Science & ML
        "ml": [
            "machine learning", "ml", "supervised learning", "unsupervised learning",
            "classification", "regression", "clustering", "decision tree",
            "random forest", "svm", "support vector", "gradient descent",
            "feature engineering", "cross validation", "overfitting", "underfitting"
        ],
        "deep_learning": [
            "deep learning", "neural network", "cnn", "rnn", "lstm", "transformer",
            "attention mechanism", "backpropagation", "activation function",
            "dropout", "batch normalization", "convolutional"
        ],
        "tensorflow": [
            "tensorflow", "tf", "keras", "tf.keras"
        ],
        "pytorch": [
            "pytorch", "torch", "torchvision"
        ],
        "nlp": [
            "nlp", "natural language processing", "tokenization", "stemming",
            "lemmatization", "word embedding", "word2vec", "bert", "gpt",
            "transformer nlp", "sentiment analysis", "ner", "named entity"
        ],
        "data_science": [
            "data science", "data analysis", "data visualization", "eda",
            "exploratory data", "statistical", "hypothesis testing"
        ],
        
        # AI & LLM
        "ai": [
            "artificial intelligence", "ai", "generative ai", "gen ai"
        ],
        "llm": [
            "llm", "large language model", "gpt", "chatgpt", "claude", "llama",
            "prompt engineering", "rag", "retrieval augmented", "fine-tuning",
            "langchain", "vector database", "embedding"
        ],
        
        # Databases
        "sql": [
            "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
            "join sql", "query sql", "index sql", "transaction", "acid"
        ],
        "nosql": [
            "nosql", "mongodb", "redis", "cassandra", "dynamodb", "couchdb",
            "document database", "key-value store"
        ],
        
        # DevOps & Cloud
        "docker": [
            "docker", "container", "dockerfile", "docker-compose", "kubernetes",
            "k8s", "pod", "deployment kubernetes"
        ],
        "aws": [
            "aws", "amazon web services", "ec2", "s3", "lambda aws", "rds",
            "cloudformation", "api gateway"
        ],
        "azure": [
            "azure", "microsoft azure", "azure functions"
        ],
        "gcp": [
            "gcp", "google cloud", "bigquery", "cloud functions gcp"
        ],
        "git": [
            "git", "github", "gitlab", "version control", "branch", "merge",
            "pull request", "commit"
        ],
        "linux": [
            "linux", "ubuntu", "bash", "shell script", "command line"
        ],
    }
    
    # Domain mapping based on technology
    TECHNOLOGY_TO_DOMAIN: Dict[str, str] = {
        # Frontend
        "react": "frontend",
        "angular": "frontend",
        "vue": "frontend",
        "javascript": "frontend",
        "typescript": "frontend",
        "html": "frontend",
        "css": "frontend",
        
        # Backend
        "python": "backend",
        "java": "backend",
        "nodejs": "backend",
        "go": "backend",
        "rust": "backend",
        "csharp": "backend",
        "php": "backend",
        "ruby": "backend",
        
        # Data Science & ML
        "ml": "data-science",
        "deep_learning": "data-science",
        "tensorflow": "data-science",
        "pytorch": "data-science",
        "nlp": "data-science",
        "data_science": "data-science",
        
        # AI
        "ai": "ai",
        "llm": "ai",
        
        # Databases
        "sql": "database",
        "nosql": "database",
        
        # DevOps
        "docker": "devops",
        "aws": "devops",
        "azure": "devops",
        "gcp": "devops",
        "git": "devops",
        "linux": "devops",
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the query classifier. Uses LLM when api_key is available."""
        # Pre-compile regex patterns for keyword fallback
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for tech, keywords in self.TECHNOLOGY_KEYWORDS.items():
            patterns = []
            for kw in keywords:
                pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                patterns.append(pattern)
            self._compiled_patterns[tech] = patterns

        try:
            from app.config import settings, get_completion_client_config

            if api_key:
                self._api_key = api_key
                self._model = model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
                self._client = OpenAI(api_key=self._api_key)
            else:
                key, default_model, base_url = get_completion_client_config()
                self._api_key = key
                self._model = model or default_model
                opts = {"api_key": self._api_key}
                if base_url:
                    opts["base_url"] = base_url
                self._client = OpenAI(**opts)
        except Exception as ex:
            logger.warning("QueryClassifier LLM client init failed (%s); keyword fallback only", ex)
            self._api_key = api_key
            self._model = model or "gpt-4o-mini"
            self._client = OpenAI(api_key=api_key) if api_key else None
        logger.info("QueryClassifier initialized (LLM=%s, keyword fallback enabled)", "on" if self._client else "off")

    def _classify_with_llm(
        self,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Use LLM to detect which technologies the question relates to.
        Returns (technologies, domain) or ([], None) on failure or when none detected.
        """
        if not self._client or not query or not query.strip():
            return [], None

        valid = list(self.TECHNOLOGY_TO_DOMAIN.keys())
        valid_str = ", ".join(sorted(valid))

        context_block = ""
        if conversation_context and conversation_context.strip():
            context_block = f"""
Recent conversation (for context only):
{conversation_context.strip()[:800]}

"""

        prompt = f"""You are a scope classifier for a document Q&A system. Given a user question (and optional conversation context), determine which technologies or topics the question relates to.

Reply with ONLY a comma-separated list of technology names from this exact list (use these slugs only):
{valid_str}

If the question clearly relates to ONE technology, return that one. If it compares or mentions MULTIPLE (e.g. "difference between hashmap and dictionary", "Java vs Python"), return all relevant ones.
Examples:
- "what is hashmap" -> java
- "what is dictionary in python" -> python
- "difference between hashmap and dictionary" -> java, python
- "what is useState" -> react
- "explain machine learning" -> ml
- "what about that?" (when context was about React) -> react

If the question does not relate to any of the listed technologies, reply with: none

{context_block}User question: {query.strip()}

Your reply (comma-separated list or "none"):"""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            text = (response.choices[0].message.content or "").strip().lower()
        except Exception as e:
            logger.warning("QueryClassifier LLM call failed: %s", e)
            return [], None

        if not text or text == "none":
            logger.info("QueryClassifier LLM returned no technologies")
            return [], None

        # Parse: split by comma/semicolon, strip, normalize to our slugs
        parts = re.split(r"[\s,;]+", text)
        seen: set = set()
        technologies: List[str] = []
        for p in parts:
            p = p.strip().strip(".").lower()
            if not p or p == "none":
                continue
            for tech in self.TECHNOLOGY_TO_DOMAIN:
                if tech.lower() == p and tech not in seen:
                    seen.add(tech)
                    technologies.append(tech)
                    break

        if not technologies:
            logger.info("QueryClassifier LLM parse yielded no valid technologies: %s", repr(text))
            return [], None

        domain = self.TECHNOLOGY_TO_DOMAIN.get(technologies[0], "general") if len(technologies) == 1 else None
        logger.info("QueryClassifier LLM: technologies=%s domain=%s", technologies, domain)
        return technologies, domain

    def _classify_with_keywords(self, query: str) -> Tuple[List[str], Optional[str]]:
        """Keyword-based fallback when LLM is off or returns nothing."""
        if not query or not query.strip():
            return [], None
        query_lower = query.lower()
        tech_scores: Dict[str, int] = {}
        for tech, patterns in self._compiled_patterns.items():
            score = sum(len(p.findall(query_lower)) for p in patterns)
            if score > 0:
                tech_scores[tech] = score
        if not tech_scores:
            return [], None
        technologies = list(tech_scores.keys())
        domain = self.TECHNOLOGY_TO_DOMAIN.get(technologies[0], "general") if len(technologies) == 1 else None
        logger.info("QueryClassifier keyword fallback: technologies=%s domain=%s", technologies, domain)
        return technologies, domain

    def classify(self, query: str) -> Tuple[List[str], Optional[str]]:
        """
        Classify a user query to determine scope (technologies) and domain.
        Uses LLM first for accuracy; falls back to keyword-based if LLM fails or returns nothing.
        Returns all detected technologies for balanced retrieval when multiple (e.g. java, python).
        """
        if not query or not query.strip():
            return [], None

        # Try LLM first
        technologies, domain = self._classify_with_llm(query)
        if technologies:
            return technologies, domain

        # Fallback to keyword-based
        return self._classify_with_keywords(query)
    
    def get_all_technologies(self) -> List[str]:
        """Get list of all supported technologies."""
        return list(self.TECHNOLOGY_KEYWORDS.keys())
    
    def get_all_domains(self) -> List[str]:
        """Get list of all supported domains."""
        return list(set(self.TECHNOLOGY_TO_DOMAIN.values()))
    
    def get_technologies_for_domain(self, domain: str) -> List[str]:
        """Get all technologies belonging to a specific domain."""
        return [
            tech for tech, dom in self.TECHNOLOGY_TO_DOMAIN.items()
            if dom == domain
        ]

    # Short follow-up phrases where using previous-turn tech is appropriate
    _FOLLOWUP_PHRASES = (
        "what about", "and?", "explain more", "more", "same", "same in",
        "what about that", "how about", "tell me more", "go on", "continue",
    )

    def classify_with_context(
        self,
        query: str,
        recent_messages: List[Dict],
    ) -> Tuple[List[str], Optional[str], bool]:
        """
        Classify query using current text first; if no technology/domain detected,
        use conversation context (for short/follow-up queries) so LLM can infer scope.
        Returns (technologies, domain, from_context).
        """
        technologies, domain = self.classify(query)
        if technologies:
            return technologies, domain, False

        if not recent_messages or len(recent_messages) < 2:
            return [], None, False

        # Use ONLY the last user message as context (avoid assistant replies listing other topics)
        previous_messages = recent_messages[:-1]
        previous_user_texts = [
            (m.get("content") or "").strip()
            for m in previous_messages
            if (m.get("role") == "user" and (m.get("content") or "").strip())
        ]
        if not previous_user_texts:
            return [], None, False
        previous_text = previous_user_texts[-1]

        # Try LLM with conversation context so it can infer scope (e.g. "what is that?" -> java)
        if self._client:
            technologies, domain = self._classify_with_llm(query, conversation_context=previous_text)
            if technologies:
                logger.info(
                    "Query classification from conversation context (LLM): technologies=%s domain=%s",
                    technologies, domain,
                )
                return technologies, domain, True
        # Fallback: classify previous turn (keyword-based)
        technologies, domain = self._classify_with_keywords(previous_text)
        if technologies:
            logger.info(
                "Query classification from conversation context (keyword): technologies=%s domain=%s (previous_user_text=%s...)",
                technologies, domain, previous_text[:80],
            )
            return technologies, domain, True
        return [], None, False


# Global singleton instance
_classifier: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """Get or create the global query classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier


def classify_query(query: str) -> Tuple[List[str], Optional[str]]:
    """
    Convenience function to classify a query.

    Args:
        query: User's question/query string

    Returns:
        Tuple of (technologies, domain). technologies is a list (0, 1, or more).
    """
    classifier = get_query_classifier()
    return classifier.classify(query)


def classify_query_with_context(
    query: str,
    recent_messages: List[Dict],
) -> Tuple[List[str], Optional[str], bool]:
    """
    Classify query with conversation context: use current query first;
    if no technology/domain detected, derive from previous messages.

    Args:
        query: Current user query.
        recent_messages: Recent messages in chronological order.

    Returns:
        Tuple of (technologies, domain, from_context). technologies is a list.
    """
    classifier = get_query_classifier()
    return classifier.classify_with_context(query, recent_messages)
