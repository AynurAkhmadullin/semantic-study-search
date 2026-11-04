"""Веб-интерфейс общей библиотеки материалов."""

import os
import re
from functools import partial

import streamlit as st

from semantic_study_search.api.schemas import SearchResponse
from semantic_study_search.limits import MAX_QUERY_CHARACTERS
from semantic_study_search.sources import uploaded_source_id
from semantic_study_search.ui import client


def escape_markdown(text: str) -> str:
    """Экранирует разметку в текстах документов и пользовательских запросах.

    Args:
        text: Текст для показа в интерфейсе.

    Returns:
        Текст без активной Markdown-разметки.
    """
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>$])", r"\\\1", text)


def show_results(response: SearchResponse, api_url: str) -> None:
    """Показывает фрагменты и ссылки на исходные материалы.

    Args:
        response: Результаты поискового API.
        api_url: Адрес API для скачивания загруженных оригиналов.
    """
    if not response.results:
        st.info(
            "Достаточно близкие фрагменты не найдены. Попробуйте уточнить вопрос "
            "или выбрать тему из библиотеки."
        )
        return
    st.subheader(f"Найденные фрагменты · {len(response.results)}")
    st.caption("По вопросу: " + escape_markdown(response.query))
    for number, result in enumerate(response.results, 1):
        with st.container(border=True):
            st.subheader(f"{number}. {escape_markdown(result.chunk.title)}", anchor=False)
            st.caption(f"Сходство: {result.score:.3f}")
            text = result.chunk.text
            if len(text) > 600:
                preview = text[:600].rsplit(" ", 1)[0] + "…"
                st.markdown(escape_markdown(preview))
                with st.expander("Показать весь фрагмент"):
                    st.markdown(escape_markdown(text))
            else:
                st.markdown(escape_markdown(text))
            source = str(result.chunk.source_url)
            if uploaded_source_id(source) is not None:
                st.download_button(
                    "Скачать материал",
                    data=partial(client.download_material, api_url, source),
                    file_name=result.chunk.file_name or "material",
                    key=f"download-{result.chunk.id}",
                    on_click="ignore",
                    icon=":material/download:",
                )
            else:
                st.link_button("Открыть источник", source, icon=":material/open_in_new:")
    st.caption(
        "Сходство отражает близость текстов по смыслу и не является вероятностью правильного ответа."
    )


@st.fragment(run_every=1)
def upload_progress(api_url: str) -> None:
    """Обновляет состояние загрузки и активирует поиск после готовности.

    Args:
        api_url: Адрес API загрузок.
    """
    identifier = st.session_state.get("upload_job_id")
    if not identifier:
        return
    try:
        material = client.get_material(api_url, identifier)
    except client.APIClientError as error:
        st.warning(escape_markdown(str(error)))
        return
    if material.status in {"ready", "failed"}:
        st.session_state["upload_notice"] = material.model_dump()
        st.session_state.pop("upload_job_id", None)
        st.session_state.pop("search_response", None)
        st.rerun()
    stages = {
        "queued": (0.05, "Материал ожидает обработки"),
        "extracting": (0.2, "Извлекаем текст"),
        "chunking": (0.4, "Подготавливаем фрагменты"),
        "embedding": (0.65, "Создаём поисковый индекс"),
        "saving": (0.9, "Сохраняем материал"),
    }
    progress, label = stages[material.status]
    st.progress(progress, text=label)
    st.caption("После завершения материал появится в поиске автоматически.")


def show_materials(api_url: str) -> None:
    """Показывает загрузки, скачивание и удаление с подтверждением.

    Args:
        api_url: Адрес API общей библиотеки.
    """
    try:
        materials = client.list_materials(api_url).materials
    except client.APIClientError as error:
        st.error(escape_markdown(str(error)))
        return
    if not materials:
        st.session_state.pop("delete_material_id", None)
        st.info("Материалы ещё не загружены.")
        return
    processing = any(material.status not in {"ready", "failed"} for material in materials)
    for material in materials:
        st.write(escape_markdown(material.title))
        st.caption(escape_markdown(material.file_name))
        if material.status == "ready":
            st.caption(f"Готово · фрагментов: {material.fragments}")
            st.download_button(
                "Скачать оригинал",
                data=partial(client.download_material, api_url, f"/materials/{material.id}/file"),
                file_name=material.file_name,
                key=f"original-{material.id}",
                on_click="ignore",
            )
        elif material.status == "failed":
            st.error(escape_markdown(material.error or "Ошибка обработки."))
        else:
            st.caption("Материал обрабатывается.")
            if st.session_state.get("upload_job_id") != material.id:
                st.session_state["upload_job_id"] = material.id
                st.rerun()
        if st.button(
            "Удалить материал",
            key=f"delete-{material.id}",
            disabled=processing,
            icon=":material/delete:",
        ):
            st.session_state["delete_material_id"] = material.id
        if st.session_state.get("delete_material_id") != material.id:
            continue
        st.warning(
            f"Удалить «{escape_markdown(material.title)}»? "
            "Файл и его фрагменты будут удалены из общей библиотеки для всех пользователей."
        )
        confirm, cancel = st.columns(2)
        with cancel:
            if st.button("Отмена", key=f"cancel-delete-{material.id}"):
                st.session_state.pop("delete_material_id", None)
                st.rerun()
        with confirm:
            if st.button(
                "Удалить окончательно",
                key=f"confirm-delete-{material.id}",
                type="primary",
                disabled=processing,
            ):
                try:
                    client.delete_material(api_url, material.id)
                except client.APIClientError as error:
                    st.error(escape_markdown(str(error)))
                else:
                    for key in ("delete_material_id", "search_response", "search_error"):
                        st.session_state.pop(key, None)
                    if st.session_state.get("upload_notice", {}).get("id") == material.id:
                        st.session_state.pop("upload_notice", None)
                    st.session_state["delete_notice"] = f"Материал «{material.title}» удалён."
                    st.rerun()


def show_upload(api_url: str) -> None:
    """Показывает форму загрузки, ход подготовки и историю материалов.

    Args:
        api_url: Адрес API загрузок.
    """
    with st.expander(
        "Добавить материал",
        expanded=bool(
            st.session_state.get("upload_job_id")
            or st.session_state.get("upload_notice")
            or st.session_state.get("show_materials")
        ),
    ):
        st.caption(
            "PDF с текстовым слоем, TXT или Markdown — до 20 МБ. Материал станет доступен всем пользователям библиотеки."
        )
        with st.form("upload_form", clear_on_submit=True):
            file = st.file_uploader(
                "Материал", type=["pdf", "txt", "md", "markdown"], max_upload_size=20
            )
            title = st.text_input("Название материала (необязательно)", max_chars=200)
            submitted = st.form_submit_button(
                "Загрузить материал", disabled=bool(st.session_state.get("upload_job_id"))
            )
        if submitted:
            if file is None:
                st.error("Выберите файл для загрузки.")
            else:
                try:
                    material = client.upload_material(api_url, file.name, file.getvalue(), title)
                    st.session_state["upload_job_id"] = material.id
                    st.session_state.pop("upload_notice", None)
                except client.APIClientError as error:
                    st.error(escape_markdown(str(error)))
        if "upload_job_id" in st.session_state:
            upload_progress(api_url)
        if notice := st.session_state.get("upload_notice"):
            if notice["status"] == "ready":
                st.success(
                    f"Материал «{escape_markdown(notice['title'])}» готов к поиску. Фрагментов: {notice['fragments']}."
                )
            else:
                st.error(escape_markdown(notice["error"] or "Не удалось подготовить материал."))
            for warning in notice["warnings"]:
                st.warning(escape_markdown(warning))
        if notice := st.session_state.pop("delete_notice", None):
            st.success(escape_markdown(notice))
        if st.button("Загрузки"):
            st.session_state["show_materials"] = True
        if st.session_state.get("show_materials"):
            show_materials(api_url)


def main() -> None:
    """Отображает форму поиска и обрабатывает ответы поискового API."""
    st.set_page_config(
        page_title="Поиск по материалам", page_icon=":material/menu_book:", layout="wide"
    )
    st.markdown(
        "<style>.block-container{max-width:1080px;padding-top:2.5rem;}"
        "[data-testid='stForm']{background:white;border-radius:16px;}"
        "[data-testid='stVerticalBlockBorderWrapper']{border-radius:16px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    api_url = os.environ.get("SSS_API_URL", "http://127.0.0.1:8000").strip()
    search_available = False
    library_empty = False
    with st.sidebar:
        st.header("Библиотека", anchor=False)
        try:
            health = client.get_health(api_url)
            if health.status == "ready":
                library_empty = health.documents == 0
                search_available = not library_empty
                if search_available:
                    st.success("Поиск доступен", icon=":material/check_circle:")
                left, right = st.columns(2)
                left.metric("Документы", health.documents)
                right.metric("Фрагменты", health.fragments)
                if health.documents == 0:
                    st.info("Библиотека пуста. Добавьте первый материал.")
            else:
                st.warning("Поиск временно недоступен.")
        except client.APIClientError:
            st.warning("Нет связи с поисковым сервисом.")
        st.button("Обновить состояние", icon=":material/refresh:")
        st.divider()
        st.markdown("**Как искать**")
        st.write(
            "Задайте вопрос своими словами. Откройте найденный источник, чтобы прочитать объяснение в контексте."
        )

    st.caption("SEMANTIC STUDY SEARCH")
    st.title("Найдите нужное в материалах")
    st.write(
        "Задайте вопрос — найдём близкие по смыслу фрагменты и покажем, где прочитать подробнее."
    )
    if library_empty:
        st.info("Сначала добавьте материал ниже. Поиск станет доступен после его обработки.")
    if not search_available:
        st.session_state.pop("search_response", None)
        st.session_state.pop("search_error", None)
    with st.form("search_form"):
        st.text_input(
            "Ваш вопрос",
            key="query_input",
            max_chars=MAX_QUERY_CHARACTERS,
            placeholder="Введите вопрос по загруженным материалам",
            disabled=not search_available,
        )
        relaxed = st.checkbox(
            "Показать и менее точные совпадения",
            help="Расширяет выдачу, если по новой теме ничего не находится. Фрагменты могут не отвечать на вопрос.",
            disabled=not search_available,
        )
        count, action = st.columns([1, 2], vertical_alignment="bottom")
        with count:
            top_k = st.selectbox(
                "Количество фрагментов",
                options=[3, 5, 10, 20],
                index=1,
                disabled=not search_available,
            )
        with action:
            submitted = st.form_submit_button(
                "Найти",
                type="primary",
                icon=":material/search:",
                width="stretch",
                disabled=not search_available,
            )

    show_upload(api_url)

    if submitted and search_available:
        st.session_state.pop("search_response", None)
        st.session_state.pop("search_error", None)
        query = st.session_state.get("query_input", "").strip()
        if not query:
            st.session_state["search_error"] = "Введите вопрос для поиска."
        else:
            try:
                with st.spinner("Ищем подходящие фрагменты…"):
                    response = (
                        client.search(api_url, query, top_k, relaxed=True)
                        if relaxed
                        else client.search(api_url, query, top_k)
                    )
                st.session_state["search_response"] = response.model_dump(mode="json")
            except client.APIClientError as error:
                st.session_state["search_error"] = str(error)
    if st.session_state.get("search_error"):
        st.error(escape_markdown(st.session_state["search_error"]))
    if "search_response" in st.session_state:
        show_results(SearchResponse.model_validate(st.session_state["search_response"]), api_url)


if __name__ == "__main__":
    main()
