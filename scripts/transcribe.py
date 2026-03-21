"""Транскрибация аудио через faster-whisper на V100.

Использование:
    python scripts/transcribe.py datasets/yandex_conf.mp3 --start 1320 --output datasets/yandex_rag_transcript.txt

    --start: секунда начала (22:00 = 1320)
    --end: секунда конца (опционально)
    --model: размер модели (default: large-v3)
    --language: язык (default: ru)
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Транскрибация аудио через faster-whisper")
    parser.add_argument("audio", help="Путь к аудио файлу")
    parser.add_argument("--start", type=int, default=0, help="Начало в секундах (22:00 = 1320)")
    parser.add_argument("--end", type=int, default=None, help="Конец в секундах")
    parser.add_argument("--model", default="large-v3", help="Модель whisper (default: large-v3)")
    parser.add_argument("--language", default="ru", help="Язык (default: ru)")
    parser.add_argument("--output", default=None, help="Файл для сохранения текста")
    parser.add_argument("--device", default="cuda", help="cuda или cpu")
    parser.add_argument("--compute-type", default="float16", help="float16, int8_float16, int8")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Установи faster-whisper: pip install faster-whisper")
        sys.exit(1)

    print(f"Загрузка модели {args.model} на {args.device}...")
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"Модель загружена за {time.time() - t0:.1f}с")

    print(f"Транскрибация {args.audio} (язык: {args.language})...")
    if args.start > 0:
        print(f"  Начало: {args.start // 60}:{args.start % 60:02d}")
    if args.end:
        print(f"  Конец: {args.end // 60}:{args.end % 60:02d}")

    t0 = time.time()
    segments, info = model.transcribe(
        args.audio,
        language=args.language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    lines = []
    # Инкрементальная запись в файл (защита от segfault)
    out_path = args.output or (args.audio.rsplit(".", 1)[0] + "_transcript.txt")
    out_file = open(out_path, "w", encoding="utf-8")

    for segment in segments:
        # Фильтруем по таймкодам
        if segment.start < args.start:
            continue
        if args.end and segment.start > args.end:
            break

        timestamp = f"[{int(segment.start // 60)}:{int(segment.start % 60):02d}]"
        line = f"{timestamp} {segment.text.strip()}"
        lines.append(line)
        out_file.write(line + "\n")
        out_file.flush()

    out_file.close()
    elapsed = time.time() - t0
    print(f"\nТранскрибация завершена: {len(lines)} сегментов за {elapsed:.1f}с")
    print(f"Detected language: {info.language} (prob={info.language_probability:.2f})")
    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    main()
