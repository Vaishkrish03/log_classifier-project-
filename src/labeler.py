"""
labeler.py
----------
Assigns multi-class labels using Context-Aware Heuristics.
Uses Regex boundaries and negations to avoid false positive biases.
"""
import re

CRITICAL_PATTERNS = [
    r"\b(fatal|critical|panic|segfault|oom|out of memory|crash|corrupt|halted|nullpointer|exception|abort|killed|denied|refused|unreachable|deadlock|traceback|stacktrace|core dumped|unhandled|access violation|segmentation fault|kernel panic|offline|broken|invalid|illegal|breach|vulnerability|unauthorized|missing|cannot|unable|down|lost)\b",
    r"(?<!0\s)\b(error|errors|failed|failure)\b(?!\s*:?\s*0)",
    r"disk full",
]

WARNING_PATTERNS = [
    r"\b(warn|warning|deprecated|degraded|fallback)\b(?!\s*:?\s*0)",
    r"\b(retry|retrying|slow|exceed|threshold|limit|delay|skipped|mismatch|overflow|disconnect|disconnected|lag|latency|dropped|bottleneck|starvation|throttled|unauthenticated|depleted|high|skip|partial)\b",
]

INFO_PATTERNS = [
    r"\b(info|started|starting|initialized|connected|connecting|loaded|loading|registered|success|completed|received|sent|processing|created|updated|running|listening|ready|enabled|configured|detected|found|allocated)\b",
]

NORMAL_PATTERNS = [
    r"\b(debug|trace|verbose|heartbeat|ping|pong|idle|alive|healthy|normal|routine|scheduled|checkpoint|status|report|metric|log|audit|periodic|flush)\b",
    r"(?<!no\s)\b(ok)\b",
    r"status\s*:?\s*(ok|200|normal)",
]

CLASS_NAMES = {0: "Critical", 1: "Warning", 2: "Info", 3: "Normal"}

TIE_PRIORITY = [1, 0, 2, 3]


def _score_patterns(line: str, patterns: list) -> int:
    line_lower = line.lower()
    total = 0
    for pattern in patterns:
        matches = re.findall(pattern, line_lower)
        total += len(matches) * 2
    return total


def label_log_multi(line: str) -> int:
    # Split camelCase so OutOfMemoryError → Out Of Memory Error
    line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)

    c = _score_patterns(line, CRITICAL_PATTERNS)
    w = _score_patterns(line, WARNING_PATTERNS)
    i = _score_patterns(line, INFO_PATTERNS)
    n = _score_patterns(line, NORMAL_PATTERNS)

    scores = {0: c, 1: w, 2: i, 3: n}
    max_score = max(scores.values())

    if max_score == 0:
        return 3

    tied = [cls for cls, score in scores.items() if score == max_score]

    if len(tied) == 1:
        return tied[0]

    for priority_class in TIE_PRIORITY:
        if priority_class in tied:
            return priority_class


def label_log_name(line: str) -> str:
    return CLASS_NAMES[label_log_multi(line)]
    return CLASS_NAMES[label_log_multi(line)]
