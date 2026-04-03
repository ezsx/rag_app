"""Unit-тесты для smart chunking в scripts/ingest_telegram.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_ingest_module():
    """Загружает scripts/ingest_telegram.py как обычный модуль."""
    spec = importlib.util.spec_from_file_location(
        "ingest_telegram_module",
        Path(__file__).parent.parent.parent / "scripts" / "ingest_telegram.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_short_post_no_chunking():
    """Короткий пост должен возвращаться одним чанком."""
    mod = _load_ingest_module()
    text = "Короткий пост про LLM."
    chunks = mod._smart_chunk(text, threshold=1500, target=1200)
    assert chunks == [text]


def test_long_post_splits_by_paragraphs():
    """Длинный пост с абзацами должен делиться на несколько чанков."""
    mod = _load_ingest_module()
    paragraph = "A" * 900
    text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"
    chunks = mod._smart_chunk(text, threshold=1500, target=1200)
    assert len(chunks) >= 2


def test_chunk_target_size():
    """Каждый чанк должен быть не длиннее target."""
    mod = _load_ingest_module()
    text = ("B" * 500 + "\n") * 8
    chunks = mod._smart_chunk(text, threshold=1500, target=1200)
    assert chunks
    assert all(len(chunk) <= 1200 for chunk in chunks)


def test_no_empty_chunks():
    """После recursive split не должно оставаться пустых чанков."""
    mod = _load_ingest_module()
    text = ("C" * 800) + "\n\n\n\n" + ("D" * 800) + "\n\n" + ("E" * 800)
    chunks = mod._smart_chunk(text, threshold=1500, target=1200)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_single_long_paragraph_hard_split():
    """Один длинный блок без абзацев должен делиться по более мелким сепараторам."""
    mod = _load_ingest_module()
    sentence = "Это длинное предложение про модели и retrieval. "
    text = sentence * 80
    chunks = mod._smart_chunk(text, threshold=1500, target=1200)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 1200 for chunk in chunks)
