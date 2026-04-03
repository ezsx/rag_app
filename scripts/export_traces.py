"""
Экспорт трейсов из Langfuse в чистый JSON с деревом спанов.

Использование:
  # Экспорт последних 17 трейсов
  python scripts/export_traces.py --limit 17

  # Экспорт конкретных трейсов по ID
  python scripts/export_traces.py --ids "trace-id-1,trace-id-2,trace-id-3"

  # Фильтр по тегу
  python scripts/export_traces.py --limit 17 --tag production

  # Фильтр по имени трейса
  python scripts/export_traces.py --limit 17 --name agent_request

  # Фильтр по времени (ISO 8601)
  python scripts/export_traces.py --limit 17 --from-time "2026-04-03T13:19:00Z" --to-time "2026-04-03T13:30:00Z"

Переменные окружения:
  LANGFUSE_SECRET_KEY  — секретный ключ (sk-lf-...)
  LANGFUSE_PUBLIC_KEY  — публичный ключ (pk-lf-...)
  LANGFUSE_HOST        — URL инстанса (по умолчанию https://cloud.langfuse.com)
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    from langfuse import Langfuse
except ImportError:
    print("Установи langfuse: pip install langfuse")
    sys.exit(1)


def fetch_traces(langfuse: Langfuse, args) -> list:
    """Получить список трейсов."""
    if args.ids:
        trace_ids = [tid.strip() for tid in args.ids.split(",")]
        traces = []
        for tid in trace_ids:
            try:
                t = langfuse.api.trace.get(tid)
                traces.append(t)
            except Exception as e:
                print(f"  ⚠ Трейс {tid} не найден: {e}", file=sys.stderr)
        return traces

    kwargs = {"limit": args.limit}
    if args.tag:
        kwargs["tags"] = [args.tag]
    if args.name:
        kwargs["name"] = args.name
    if args.from_time:
        kwargs["from_timestamp"] = datetime.fromisoformat(args.from_time.replace("Z", "+00:00"))
    if args.to_time:
        kwargs["to_timestamp"] = datetime.fromisoformat(args.to_time.replace("Z", "+00:00"))

    result = langfuse.api.trace.list(**kwargs)
    return result.data


def fetch_observations(langfuse: Langfuse, trace_id: str) -> list:
    """Получить все observations трейса с пагинацией."""
    all_obs = []
    page = 1
    while True:
        result = langfuse.api.observations.get_many(
            trace_id=trace_id,
            limit=100,
            page=page,
        )
        if not result.data:
            break
        all_obs.extend(result.data)
        if len(result.data) < 100:
            break
        page += 1
    return all_obs


def build_tree(observations: list) -> list:
    """Собрать плоский список observations в дерево."""
    nodes = {}
    for obs in observations:
        obs_id = obs.id
        nodes[obs_id] = {
            "name": obs.name or "(unnamed)",
            "type": getattr(obs, "type", None),
            "duration_s": round((obs.end_time - obs.start_time).total_seconds(), 2)
            if obs.start_time and obs.end_time
            else None,
            "model": getattr(obs, "model", None) or None,
            "tokens": format_tokens(obs),
            "input": simplify_io(getattr(obs, "input", None)),
            "output": simplify_io(getattr(obs, "output", None)),
            "tool_calls": extract_tool_calls(obs),
            "children": [],
            "_parent_id": getattr(obs, "parent_observation_id", None),
        }

    roots = []
    for obs_id, node in nodes.items():
        parent_id = node.pop("_parent_id")
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)

    return roots


def extract_tool_calls(obs) -> list | None:
    """Извлечь tool_calls из observation — имя + аргументы + распарсенный answer."""
    raw_calls = getattr(obs, "tool_calls", None) or []
    if not raw_calls:
        return None

    parsed = []
    for tc in raw_calls:
        try:
            tc_data = json.loads(tc) if isinstance(tc, str) else tc
            args_str = tc_data.get("arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            entry = {
                "function": tc_data.get("type", "function"),
                "name": None,
                "arguments": args,
            }
            # Имя функции может быть в разных местах
            fn = tc_data.get("function", {})
            if isinstance(fn, dict):
                entry["name"] = fn.get("name")
            if not entry["name"]:
                # Берём из toolCallNames observation
                names = getattr(obs, "tool_call_names", None) or getattr(obs, "toolCallNames", None) or []
                if names:
                    entry["name"] = names[0] if len(parsed) == 0 else names[min(len(parsed), len(names) - 1)]
            parsed.append(entry)
        except (json.JSONDecodeError, TypeError, AttributeError):
            parsed.append({"raw": str(tc)[:500]})

    return parsed if parsed else None


def format_tokens(obs) -> dict | None:
    """Извлечь токены в формат {input, output, total}."""
    usage = getattr(obs, "usage_details", None) or getattr(obs, "usage", None)
    if not usage:
        return None

    if isinstance(usage, dict):
        inp = usage.get("input", 0) or usage.get("prompt_tokens", 0) or 0
        out = usage.get("output", 0) or usage.get("completion_tokens", 0) or 0
        total = usage.get("total", 0) or usage.get("total_tokens", 0) or (inp + out)
    else:
        inp = getattr(usage, "input", None) or getattr(usage, "prompt_tokens", None) or 0
        out = getattr(usage, "output", None) or getattr(usage, "completion_tokens", None) or 0
        total = getattr(usage, "total", None) or getattr(usage, "total_tokens", None) or (inp + out)

    if total == 0:
        return None

    return {"input": inp, "output": out, "total": total}


def simplify_io(data):
    """Упростить input/output — убрать лишнюю вложенность."""
    if data is None:
        return None

    if isinstance(data, list):
        simplified = []
        for item in data:
            if isinstance(item, dict) and "role" in item and "content" in item:
                entry = {"role": item["role"], "content": item["content"]}
                if "tool_calls" in item and item["tool_calls"]:
                    entry["tool_calls"] = item["tool_calls"]
                simplified.append(entry)
            else:
                simplified.append(item)
        return simplified

    if isinstance(data, dict):
        if list(data.keys()) == ["messages"]:
            return simplify_io(data["messages"])
        if "choices" in data:
            choices = data["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message", {})
                return msg.get("content", data)
        return data

    return data


def clean_none(obj):
    """Убрать поля с None и пустые children."""
    if isinstance(obj, dict):
        return {
            k: clean_none(v)
            for k, v in obj.items()
            if v is not None and v != [] and v != {}
        }
    if isinstance(obj, list):
        return [clean_none(item) for item in obj]
    return obj


def format_trace(trace, observations: list) -> dict:
    """Сформировать чистый JSON одного трейса."""
    tree = build_tree(observations)

    result = {
        "trace_id": trace.id,
        "name": trace.name or "(unnamed)",
        "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
        "duration_s": None,
        "user_id": getattr(trace, "user_id", None),
        "session_id": getattr(trace, "session_id", None),
        "tags": getattr(trace, "tags", None) or None,
        "input": simplify_io(getattr(trace, "input", None)),
        "output": simplify_io(getattr(trace, "output", None)),
        "spans": tree,
    }

    if hasattr(trace, "latency") and trace.latency is not None:
        result["duration_s"] = round(trace.latency, 2)

    return clean_none(result)


def strip_io(obj):
    """Рекурсивно убрать input/output из всех уровней."""
    if isinstance(obj, dict):
        return {
            k: strip_io(v) for k, v in obj.items()
            if k not in ("input", "output")
        }
    if isinstance(obj, list):
        return [strip_io(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Экспорт трейсов из Langfuse")
    parser.add_argument("--limit", type=int, default=17, help="Количество трейсов")
    parser.add_argument("--ids", type=str, help="ID трейсов через запятую")
    parser.add_argument("--tag", type=str, help="Фильтр по тегу")
    parser.add_argument("--name", type=str, help="Фильтр по имени трейса")
    parser.add_argument("--from-time", type=str, help="Начало временного окна (ISO 8601)")
    parser.add_argument("--to-time", type=str, help="Конец временного окна (ISO 8601)")
    parser.add_argument("--output", "-o", type=str, default="traces_export.json", help="Файл для сохранения")
    parser.add_argument("--no-io", action="store_true", help="Не включать input/output")
    args = parser.parse_args()

    langfuse = Langfuse()

    print(f"Загружаю трейсы...")
    traces = fetch_traces(langfuse, args)
    print(f"Найдено трейсов: {len(traces)}")

    exported = []
    for i, trace in enumerate(traces):
        print(f"  [{i+1}/{len(traces)}] {trace.name or trace.id}")
        observations = fetch_observations(langfuse, trace.id)
        formatted = format_trace(trace, observations)

        if args.no_io:
            formatted = strip_io(formatted)

        exported.append(formatted)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nГотово! Сохранено в {args.output}")
    print(f"Трейсов: {len(exported)}")


if __name__ == "__main__":
    main()
