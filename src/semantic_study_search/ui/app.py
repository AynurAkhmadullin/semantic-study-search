"""Веб-интерфейс общей библиотеки учебных материалов."""

import os
import re

import streamlit as st

from semantic_study_search.api.schemas import SearchResponse
from semantic_study_search.limits import MAX_QUERY_CHARACTERS
from semantic_study_search.ui import client


def escape_markdown(text: str) -> str:
    """Экранирует разметку в текстах документов и пользовательских запросах.

    Args:
        text: Текст для показа в интерфейсе.

    Returns:
        Текст без активной Markdown-разметки.
    """
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>$])", r"\\\1", text)


def set_example(query: str) -> None:
    """Подставляет пример вопроса в форму поиска.

    Args:
        query: Выбранный пример вопроса.
    """
    st.session_state["query_input"] = query


def show_results(response: SearchResponse) -> None:
    """Показывает фрагменты и ссылки на исходные учебные материалы.

    Args:
        response: Результаты поискового API.
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
            st.link_button(
                "Открыть источник", str(result.chunk.source_url), icon=":material/open_in_new:"
            )
    st.caption(
        "Сходство отражает близость текстов по смыслу и не является вероятностью правильного ответа."
    )


def main() -> None:
    """Отображает форму поиска и обрабатывает ответы поискового API."""
    st.set_page_config(
        page_title="Поиск по учебным материалам", page_icon=":material/menu_book:", layout="wide"
    )
    st.markdown(
        "<style>.block-container{max-width:1080px;padding-top:2.5rem;}"
        "[data-testid='stForm']{background:white;border-radius:16px;}"
        "[data-testid='stVerticalBlockBorderWrapper']{border-radius:16px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    api_url = os.environ.get("SSS_API_URL", "http://127.0.0.1:8000").strip()
    with st.sidebar:
        st.header("Учебная библиотека", anchor=False)
        try:
            health = client.get_health(api_url)
            if health.status == "ready":
                st.success("Поиск доступен", icon=":material/check_circle:")
                left, right = st.columns(2)
                left.metric("Документы", health.documents)
                right.metric("Фрагменты", health.fragments)
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
        st.caption("Материалы общей библиотеки подготавливает автор проекта.")

    st.caption("SEMANTIC STUDY SEARCH")
    st.title("Найдите нужное в учебных материалах")
    st.write(
        "Задайте вопрос — найдём близкие по смыслу фрагменты и покажем, где прочитать подробнее."
    )
    with st.form("search_form"):
        st.text_input(
            "Ваш вопрос",
            key="query_input",
            max_chars=MAX_QUERY_CHARACTERS,
            placeholder="Например, зачем нужна маска внимания?",
        )
        count, action = st.columns([1, 2], vertical_alignment="bottom")
        with count:
            top_k = st.selectbox("Количество фрагментов", options=[3, 5, 10, 20], index=1)
        with action:
            submitted = st.form_submit_button(
                "Найти", type="primary", icon=":material/search:", width="stretch"
            )
    st.caption("Попробуйте один из вопросов")
    examples = [
        "Что такое токенизация?",
        "Зачем нужна маска внимания?",
        "Как сравнивают векторы текста?",
    ]
    for column, query in zip(st.columns(3), examples, strict=True):
        column.button(query, on_click=set_example, args=(query,), width="stretch")

    if submitted:
        st.session_state.pop("search_response", None)
        st.session_state.pop("search_error", None)
        query = st.session_state.get("query_input", "").strip()
        if not query:
            st.session_state["search_error"] = "Введите вопрос для поиска."
        else:
            try:
                with st.spinner("Ищем подходящие фрагменты…"):
                    response = client.search(api_url, query, top_k)
                st.session_state["search_response"] = response.model_dump(mode="json")
            except client.APIClientError as error:
                st.session_state["search_error"] = str(error)
    if st.session_state.get("search_error"):
        st.error(escape_markdown(st.session_state["search_error"]))
    if "search_response" in st.session_state:
        show_results(SearchResponse.model_validate(st.session_state["search_response"]))


if __name__ == "__main__":
    main()
