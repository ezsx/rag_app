import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from fastapi import Depends, HTTPException
import chromadb
from llama_cpp import Llama
from contextlib import asynccontextmanager

from adapters.chroma import Retriever
from services.qa_service import QAService
from services.query_planner_service import QueryPlannerService
from services.reranker_service import RerankerService
from services.agent_service import AgentService
from utils.model_downloader import auto_download_models, RECOMMENDED_MODELS
from core.settings import get_settings, Settings
from adapters.search.bm25_index import BM25IndexManager
from adapters.search.bm25_retriever import BM25Retriever
from adapters.search.hybrid_retriever import HybridRetriever
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)

# Храним последний созданный экземпляр LLM, чтобы иметь возможность освободить VRAM
_LAST_LLM_INSTANCE: Optional[Llama] = None


@lru_cache
def get_chroma_client():
    """Создает и возвращает ChromaDB клиент"""
    settings = get_settings()
    try:
        # Попробуем HTTP клиент (для Docker compose)
        client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        # Проверим подключение
        client.heartbeat()
        logger.info(
            f"Подключение к ChromaDB HTTP: {settings.chroma_host}:{settings.chroma_port}"
        )
        return client
    except Exception as e:
        logger.warning(f"HTTP подключение не удалось ({e}), пробуем локальный клиент")
        # Fallback на локальный клиент
        return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    client = get_chroma_client()
    """Создает и возвращает Retriever для поиска в ChromaDB"""
    embedding_model_key = settings.current_embedding_key
    collection_name = settings.current_collection

    # Получаем полное название модели из конфигурации
    if embedding_model_key in RECOMMENDED_MODELS["embedding"]:
        embedding_model = RECOMMENDED_MODELS["embedding"][embedding_model_key]["name"]
    else:
        # Fallback на прямое указание модели
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    logger.info(
        f"Используем embedding модель: {embedding_model} (коллекция: {collection_name})"
    )

    # Автоскачивание embedding модели если необходимо
    auto_download_embedding = (
        os.getenv("AUTO_DOWNLOAD_EMBEDDING", "true").lower() == "true"
    )
    if auto_download_embedding:
        try:
            from utils.model_downloader import download_embedding_model

            download_embedding_model(embedding_model, settings.cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось скачать embedding модель: {e}")

    return Retriever(client, collection_name, embedding_model)


@lru_cache
def get_llm():
    """Создает и возвращает LLM модель с автоскачиванием"""
    settings = get_settings()
    # Конфигурация модели из настроек
    llm_model_key = settings.current_llm_key
    models_dir = settings.models_dir
    cache_dir = settings.cache_dir
    auto_download = os.getenv("AUTO_DOWNLOAD_LLM", "true").lower() == "true"

    # Параметры модели
    n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "-1"))
    n_ctx = int(os.getenv("LLM_CONTEXT_SIZE", "8192"))
    n_threads = int(os.getenv("LLM_THREADS", "8"))
    n_batch = int(os.getenv("LLM_BATCH", "1024"))

    # Определяем путь к модели
    if llm_model_key in RECOMMENDED_MODELS["llm"]:
        model_config = RECOMMENDED_MODELS["llm"][llm_model_key]
        model_filename = model_config["filename"]
        model_path = os.path.join(models_dir, model_filename)
        logger.info(f"Используем LLM модель: {model_config['description']}")
    else:
        # Fallback на прямое указание пути
        model_path = os.getenv("LLM_MODEL_PATH", f"{models_dir}/gpt-oss-20b-Q6_K.gguf")
        logger.info(f"Используем пользовательский путь к модели: {model_path}")

    # Диагностическое логирование окружения и пути
    try:
        import llama_cpp as _ll

        logger.info(f"llama_cpp version: {getattr(_ll, '__version__', 'unknown')}")
    except Exception as _:
        logger.info("llama_cpp version: <unavailable>")
    logger.info(
        f"ENV: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
        f"LLM_GPU_LAYERS={os.getenv('LLM_GPU_LAYERS', str(n_gpu_layers))}, "
        f"LLM_CONTEXT_SIZE={os.getenv('LLM_CONTEXT_SIZE', str(n_ctx))}, "
        f"LLM_THREADS={os.getenv('LLM_THREADS', str(n_threads))}, "
        f"LLM_BATCH={os.getenv('LLM_BATCH', str(n_batch))}"
    )
    logger.info(
        f"ENV: LLM_MODEL_PATH={os.getenv('LLM_MODEL_PATH')}, MODELS_DIR={models_dir}, PWD={os.getcwd()}"
    )
    logger.info(f"Путь к модели: {model_path}")
    if os.path.exists(model_path):
        try:
            sz = Path(model_path).stat().st_size
            logger.info(f"Размер файла модели: {sz / (1024**3):.2f} GB")
        except Exception:
            pass
        # Перечислим несколько файлов в каталоге /models для верификации volume
        try:
            from itertools import islice

            entries = list(islice(Path(models_dir).glob("*"), 10))
            sample = ", ".join([e.name for e in entries])
            logger.info(f"Содержимое /models (первые 10): {sample}")
        except Exception as _e:
            logger.warning(f"Не удалось получить список /models: {_e}")

        # Проверяем доступ и магическое число GGUF + версию
        try:
            r_ok = os.access(model_path, os.R_OK)
            logger.info(f"Доступ к модели (readable): {r_ok}")
        except Exception:
            pass
        # Проверяем магическое число GGUF
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                version_bytes = f.read(4)
            try:
                import struct

                version = struct.unpack("<I", version_bytes)[0]
            except Exception:
                version = -1
            logger.info(f"GGUF header: {magic!r}, version={version}")
        except Exception as _e:
            logger.warning(f"Не удалось прочитать заголовок GGUF: {_e}")

    # Проверяем существование модели
    if not os.path.exists(model_path):
        if auto_download and llm_model_key in RECOMMENDED_MODELS["llm"]:
            logger.info(f"🔄 Модель не найдена, запускаем автоскачивание...")

            # Автоскачивание
            downloaded_path, _ = auto_download_models(
                llm_model_key=llm_model_key,
                embedding_model_key="",  # Не скачиваем embedding здесь
                models_dir=models_dir,
                cache_dir=cache_dir,
            )

            if downloaded_path and os.path.exists(downloaded_path):
                model_path = downloaded_path
                logger.info(f"✅ Модель успешно скачана: {model_path}")
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Не удалось скачать LLM модель. Проверьте подключение к интернету.",
                )
        else:
            raise FileNotFoundError(f"LLM model not found at {model_path}")

    # Загружаем модель
    try:
        # Включаем подробный лог backend и быстрый CUDA-бэкенд по умолчанию
        os.environ.setdefault("LLAMA_LOG_LEVEL", "DEBUG")
        os.environ.setdefault("GGML_VERBOSE", "1")
        os.environ.setdefault("GGML_CUDA_FORCE_CUBLAS", "1")
        logger.info(f"📚 Загружаем LLM модель: {os.path.basename(model_path)}")
        logger.info(
            f"   GPU слои: {n_gpu_layers}, Контекст: {n_ctx}, Потоки: {n_threads}, Батч: {n_batch}"
        )

        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=True,
            )
        except Exception as e1:
            logger.error(f"Первый запуск Llama не удался: {e1}. Пробуем use_mmap=False")
            try:
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    use_mmap=False,
                    verbose=True,
                )
            except Exception as e2:
                logger.error(
                    f"Повторный запуск Llama не удался: {e2}. Пробуем use_mlock=False"
                )
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    use_mmap=False,
                    use_mlock=False,
                    verbose=True,
                )

        logger.info("✅ LLM модель успешно загружена")
        # Сохраняем ссылку на последний созданный LLM
        global _LAST_LLM_INSTANCE
        _LAST_LLM_INSTANCE = llm
        return llm

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки LLM модели: {e}")
        # Дополнительные подсказки по ошибке
        if "Failed to load model from file" in str(e):
            logger.error(
                "Возможные причины: некорректный путь/имя GGUF, поврежденный файл, несовместимая квантовка."
            )
            logger.error(
                f"Проверьте наличие файла и доступ: ls -lh {models_dir}; и переменную LLM_MODEL_PATH/KEY."
            )
        raise HTTPException(
            status_code=503, detail=f"Не удалось загрузить LLM модель: {str(e)}"
        )


# Временное освобождение VRAM LLM для ingestion (при запуске задач инжеста)
@asynccontextmanager
async def release_llm_vram_temporarily():
    """
    Освобождает VRAM, занятую LLM (llama-cpp), на время блока.
    Подход: на лету выгружаем GPU-слои, устанавливая n_gpu_layers=0 через recreate;
    если это невозможно, пытаемся вызвать low-level free_buffers/flush_kv_cache.
    По завершении не загружаем модель автоматически обратно (чтобы не заново занимать VRAM).
    В сценарии QA сервис повторно создаст LLM при следующем запросе через lru_cache.clear.
    """
    # ВАЖНО: не вызываем get_llm() здесь, чтобы не инициировать загрузку модели.
    try:
        # Если модель уже была загружена — попробуем освободить ресурсы
        global _LAST_LLM_INSTANCE
        llm = _LAST_LLM_INSTANCE
        if llm is not None:
            try:
                if hasattr(llm, "flush_kv_cache"):
                    llm.flush_kv_cache()
            except Exception:
                pass
            # Попробуем инициировать GC через деструктор
            try:
                if hasattr(llm, "__del__"):
                    llm.__del__()
            except Exception:
                pass
        # Сбросим кэш фабрики LLM, чтобы при следующем QA-запросе модель создавалась заново
        try:
            get_llm.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        # Обнулим нашу ссылку — даём GC шанс освободить VRAM
        try:
            _LAST_LLM_INSTANCE = None
        except Exception:
            pass
        yield
    finally:
        # Не пересоздаём автоматически; LLM поднимется при следующем QA-запросе
        pass


@lru_cache
def get_query_planner() -> QueryPlannerService:
    settings = get_settings()
    # Используем отдельную LLM для планировщика (CPU), если указана
    try:
        planner_llm = get_planner_llm()
    except Exception:
        planner_llm = get_llm()
    return QueryPlannerService(planner_llm, settings)


@lru_cache
def get_planner_llm():
    """Создает LLM для Query Planner (как правило, CPU Qwen2.5-3B-Instruct)."""
    import os
    from llama_cpp import Llama

    settings = get_settings()
    key = settings.planner_llm_key
    models_dir = settings.models_dir
    cache_dir = settings.cache_dir

    # Автоскачивание, если в реестре
    try:
        if key in RECOMMENDED_MODELS["llm"]:
            logger.info(
                f"📥 Planner LLM: {RECOMMENDED_MODELS['llm'][key]['description']}"
            )
            from utils.model_downloader import auto_download_models

            path, _ = auto_download_models(
                llm_model_key=key,
                embedding_model_key="",
                models_dir=models_dir,
                cache_dir=cache_dir,
            )
            model_path = path or os.getenv(
                "PLANNER_LLM_MODEL_PATH",
                f"{models_dir}/{RECOMMENDED_MODELS['llm'][key]['filename']}",
            )
        else:
            model_path = os.getenv("PLANNER_LLM_MODEL_PATH", f"{models_dir}/{key}.gguf")
    except Exception as e:
        logger.warning(f"Planner LLM auto-download skipped: {e}")
        model_path = os.getenv("PLANNER_LLM_MODEL_PATH", f"{models_dir}/{key}.gguf")

    # Загружаем на CPU (без VRAM). Для Qwen2.5 используем корректный chat формат.
    chat_format = os.getenv("PLANNER_CHAT_FORMAT", "auto")
    kwargs = dict(
        model_path=model_path,
        n_gpu_layers=0,
        n_ctx=int(os.getenv("PLANNER_LLM_CONTEXT_SIZE", "2048")),
        n_threads=int(os.getenv("PLANNER_LLM_THREADS", "6")),
        n_batch=int(os.getenv("PLANNER_LLM_BATCH", "256")),
        verbose=False,
    )
    # Если явно указали формат — пробросим, иначе оставим auto (llama.cpp из GGUF)
    if chat_format and chat_format != "auto":
        kwargs["chat_format"] = chat_format

    llm = Llama(**kwargs)  # type: ignore[arg-type]
    logger.info(f"✅ Planner LLM загружен на CPU: {os.path.basename(model_path)}")
    return llm


@lru_cache
def get_bm25_index_manager() -> BM25IndexManager:
    settings = get_settings()
    return BM25IndexManager(
        index_root=settings.bm25_index_root,
        reload_min_interval_sec=settings.bm25_reload_min_interval_sec,
    )


@lru_cache
def get_qa_service() -> QAService:
    settings = get_settings()
    retriever = get_retriever()

    # Передаем фабрику вместо готовой LLM: ленивая загрузка при первом вызове
    def _llm_factory():
        return get_llm()

    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    planner = get_query_planner() if settings.enable_query_planner else None
    reranker = get_reranker() if settings.enable_reranker else None

    hybrid = None
    if settings.hybrid_enabled:
        try:
            bm25_mgr = get_bm25_index_manager()
            bm25_ret = BM25Retriever(bm25_mgr, settings)
            hybrid = HybridRetriever(bm25_ret, retriever, settings)
        except Exception as e:
            logger.warning(f"Hybrid retriever init failed: {e}")

    return QAService(
        retriever,
        _llm_factory,
        top_k,
        settings=settings,
        planner=planner,
        reranker=reranker,
        hybrid=hybrid,
    )


@lru_cache
def get_hybrid_retriever() -> Optional[HybridRetriever]:
    settings = get_settings()
    if not settings.hybrid_enabled:
        return None
    try:
        bm25_mgr = get_bm25_index_manager()
        dense_ret = get_retriever()
        bm25_ret = BM25Retriever(bm25_mgr, settings)
        return HybridRetriever(bm25_ret, dense_ret, settings)
    except Exception as e:
        logger.warning(f"Hybrid retriever init failed: {e}")
        return None


@lru_cache
def get_reranker() -> Optional[RerankerService]:
    settings = get_settings()
    if not settings.enable_reranker:
        return None
    # Автоскачивание модели ререйкера (как и для LLM/Embedding)
    auto_download_reranker = (
        os.getenv("AUTO_DOWNLOAD_RERANKER", "true").lower() == "true"
    )
    if auto_download_reranker:
        try:
            from utils.model_downloader import download_reranker_model

            download_reranker_model(settings.reranker_model_key, settings.cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось скачать reranker модель заранее: {e}")
    return RerankerService(settings.reranker_model_key)


# === Новые зависимости для кеширования ===


@lru_cache
def get_redis_client(settings: Settings = Depends(get_settings)):
    """Создает Redis клиент если включено кеширование"""
    if not settings.redis_enabled:
        return None

    try:
        import redis

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
        )
        # Проверяем подключение
        client.ping()
        logger.info(f"Redis подключен: {settings.redis_host}:{settings.redis_port}")
        return client
    except Exception as e:
        logger.warning(f"Redis недоступен: {e}")
        return None


@lru_cache
def get_agent_service() -> AgentService:
    """Создает AgentService с настроенными зависимостями"""
    settings = get_settings()

    # Создаем ToolRunner с настроенным таймаутом
    tool_runner = ToolRunner(default_timeout_sec=settings.agent_tool_timeout)

    # Регистрируем базовые инструменты
    from services.tools.router_select import router_select
    from services.tools.compose_context import compose_context
    from services.tools.fetch_docs import fetch_docs
    from services.tools.verify import verify
    from services.tools.search import search
    from services.tools.query_plan import query_plan
    from services.tools.rerank import rerank

    # Получаем retriever для инструментов, которые его используют
    retriever = get_retriever()

    # Получаем гибридный retriever для search инструмента
    from adapters.search.hybrid_retriever import HybridRetriever

    # Пытаемся взять готовый гибридный ретривер из кеша
    hybrid_retriever = get_hybrid_retriever()
    if hybrid_retriever is None:
        # Фолбэк: собираем из BM25 и dense вручную
        try:
            bm25_mgr = get_bm25_index_manager()
            bm25_ret = BM25Retriever(bm25_mgr, settings)
            dense_ret = retriever  # текущий dense Retriever из Chroma
            hybrid_retriever = HybridRetriever(bm25_ret, dense_ret, settings)
        except Exception as e:
            logger.warning(f"Hybrid retriever fallback init failed: {e}")
            hybrid_retriever = None

    # Получаем reranker для rerank инструмента
    from services.reranker_service import RerankerService

    reranker = get_reranker()

    # Получаем query planner для query_plan инструмента
    from services.query_planner_service import QueryPlannerService

    def _llm_factory():
        return get_llm()

    query_planner = get_query_planner() if settings.enable_query_planner else None

    # Создаем обертки для инструментов, которые нуждаются в зависимостях
    def verify_wrapper(**kwargs):
        return verify(retriever=retriever, **kwargs)

    def fetch_docs_wrapper(**kwargs):
        return fetch_docs(retriever=retriever, **kwargs)

    # Эксплицитные завимосити для search-фолбэков
    bm25_mgr = get_bm25_index_manager()
    bm25_ret = BM25Retriever(bm25_mgr, settings)

    def search_wrapper(**kwargs):
        return search(
            hybrid_retriever=hybrid_retriever,
            bm25_retriever=bm25_ret,
            **kwargs,
        )

    def query_plan_wrapper(**kwargs):
        return query_plan(query_planner=query_planner, **kwargs)

    def rerank_wrapper(**kwargs):
        return rerank(reranker=reranker, **kwargs)

    tool_runner.register("router_select", router_select)
    tool_runner.register("query_plan", query_plan_wrapper, timeout_sec=6.0)
    tool_runner.register(
        "search",
        search_wrapper,
        timeout_sec=max(5.0, settings.agent_tool_timeout * 0.8),
    )
    tool_runner.register(
        "rerank",
        rerank_wrapper,
        timeout_sec=max(4.0, settings.agent_tool_timeout * 0.5),
    )
    tool_runner.register("fetch_docs", fetch_docs_wrapper)
    tool_runner.register("compose_context", compose_context)
    tool_runner.register("verify", verify_wrapper)

    # Передаем фабрику LLM для ленивой загрузки
    def _llm_factory():
        return get_llm()

    # Получаем QA сервис для fallback
    qa_service = get_qa_service()

    return AgentService(
        llm_factory=_llm_factory,
        tool_runner=tool_runner,
        settings=settings,
        qa_service=qa_service,
    )
