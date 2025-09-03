"""
GBNF грамматики для строгой генерации SearchPlan и микро-массивов строк.

Используется llama.cpp (llama-cpp-python) grammar-декодирование.
"""

from typing import Optional
import hashlib
import logging

try:
    from llama_cpp import LlamaGrammar  # type: ignore
except Exception:  # pragma: no cover
    LlamaGrammar = None  # type: ignore

logger = logging.getLogger(__name__)


# Полная грамматика SearchPlan (совместимая с llama.cpp, без комментариев)
SEARCH_PLAN_GBNF: str = r"""
root ::= searchplan
searchplan ::= "{" ws "\"normalized_queries\"" ws ":" ws queriesarray ws "," "\"must_phrases\"" ws ":" ws stringarray ws "," "\"should_phrases\"" ws ":" ws stringarray ws "," "\"metadata_filters\"" ws ":" ws ( "null" | filtersobject ) ws "," "\"k_per_query\"" ws ":" ws knumber ws "," "\"fusion\"" ws ":" ws "\"" ( "rrf" | "mmr" ) "\"" ws "}"
queriesarray ::= "[" ws stringliteral "," ws stringliteral "," ws stringliteral ( "," ws stringliteral )? ( "," ws stringliteral )? ( "," ws stringliteral )? ws "]"
stringarray ::= "[" ws ( stringliteral ( "," ws stringliteral )* )? ws "]"
filtersobject ::= "{" ws filterpair ( "," ws filterpair )* ws "}"
filterpair ::= "\"channel_usernames\"" ws ":" ws stringarray | "\"channel_ids\"" ws ":" ws intarray | "\"date_from\"" ws ":" ws datestring | "\"date_to\"" ws ":" ws datestring | "\"min_views\"" ws ":" ws number | "\"reply_to\"" ws ":" ws number
intarray ::= "[" ws ( intnumber ( "," ws intnumber )* )? ws "]"
stringliteral ::= "\"" ( stringchar | escape )* "\""
stringchar ::= [^"\\]
escape ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex ::= [0-9A-Fa-f]
datestring ::= "\"" digit digit digit digit "-" digit digit "-" digit digit "\""
digit ::= [0-9]
number ::= sign? intpart frac? exp?
intnumber ::= sign? ( "0" | nonzero digit* )
knumber ::= "50" | [1-9] | [1-4][0-9]
sign ::= "-"
nonzero ::= [1-9]
intpart ::= "0" | nonzero digit*
frac ::= "." digit+
exp ::= ( "e" | "E" ) ( "+" | "-" )? digit+
ws ::= ( [ \t\n\r] )*
"""


def _norm_endl(g: str) -> str:
    return g.replace("\r\n", "\n").replace("\r", "\n")


_GBNF_CHECKED = False


def gbnf_selfcheck() -> None:
    global _GBNF_CHECKED
    if _GBNF_CHECKED:
        return
    if LlamaGrammar is None:
        raise RuntimeError("llama_cpp не установлен: LlamaGrammar недоступен")
    # 1) минимальная грамматика
    LlamaGrammar.from_string('root ::= "X"')
    # 2) базовый JSON-стринг
    LlamaGrammar.from_string(
        _norm_endl(
            r"""
root ::= s
s ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f] ) )* "\""
"""
        )
    )
    # 3) SearchPlan
    LlamaGrammar.from_string(_norm_endl(SEARCH_PLAN_GBNF))
    # 4) микро на 2
    LlamaGrammar.from_string(
        _norm_endl(
            r"""
root ::= array
array ::= "[" ws stringliteral "," ws stringliteral ws "]"
stringliteral ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f] ) )* "\""
ws ::= ( [ \t\n\r] )*
"""
        )
    )
    _GBNF_CHECKED = True


def _gbnf_dbg(label: str, g: str) -> str:
    s = _norm_endl(g)
    md5 = hashlib.md5(s.encode("utf-8")).hexdigest()
    head = s[:120].replace("\n", "\\n")
    tail = s[-120:].replace("\n", "\\n")
    logger.info("GBNF[%s]: md5=%s head='%s' ... tail='%s'", label, md5, head, tail)
    return md5


def build_searchplan_grammar():
    if LlamaGrammar is None:
        raise RuntimeError("llama_cpp не установлен: LlamaGrammar недоступен")
    s = _norm_endl(SEARCH_PLAN_GBNF)
    _gbnf_dbg("searchplan-src", s)
    gr = LlamaGrammar.from_string(s)
    logger.info("GBNF[searchplan]: object_id=%s", id(gr))
    return gr


def build_micro_grammar(n: int):
    if LlamaGrammar is None:
        raise RuntimeError("llama_cpp не установлен: LlamaGrammar недоступен")
    if n <= 0:
        raise ValueError("n must be >= 1")
    tail = ' ( "," ws stringliteral )' * (n - 1)
    seq = f"stringliteral{tail}"
    micro_gbnf = (
        "root ::= array\n"
        'array ::= "[" ws <<SEQ>> ws "]"\n'
        'stringliteral ::= "\\"" ( stringchar | escape )* "\\""\n'
        'stringchar ::= [^"\\\\]\n'
        'escape ::= "\\\\" ( ["\\\\/bfnrt] | "u" hex hex hex hex )\n'
        "hex ::= [0-9A-Fa-f]\n"
        "ws ::= ( [ \t\n\r] )*\n"
    ).replace("<<SEQ>>", seq)
    s = _norm_endl(micro_gbnf)
    _gbnf_dbg(f"micro-{n}-src", s)
    gr = LlamaGrammar.from_string(s)
    logger.info("GBNF[micro-%d]: object_id=%s", n, id(gr))
    return gr


# Публичные API (с совместимостью)


def get_searchplan_grammar():
    return build_searchplan_grammar()


def get_string_array_grammar(n: int):
    return build_micro_grammar(n)


__all__ = [
    "SEARCH_PLAN_GBNF",
    "gbnf_selfcheck",
    "build_searchplan_grammar",
    "build_micro_grammar",
    "get_searchplan_grammar",
    "get_string_array_grammar",
]
