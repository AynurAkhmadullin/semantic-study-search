"""Безопасные ссылки на оригиналы загруженных материалов."""

import re

LOCAL_SOURCE_PATTERN = r"^/materials/[0-9a-f]{32}/file(?:#page=[1-9][0-9]*)?$"


def uploaded_source_id(source: str) -> str | None:
    """Извлекает идентификатор из внутренней ссылки на файл.

    Args:
        source: Ссылка на источник фрагмента.

    Returns:
        Идентификатор материала или None для внешней ссылки.
    """
    if re.fullmatch(LOCAL_SOURCE_PATTERN, source):
        return source.split("/")[2]
    return None
