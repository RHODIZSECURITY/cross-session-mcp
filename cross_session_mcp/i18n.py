"""Language of the documents the server writes: handoffs, the resume wrapper and MEMORY.md.

`CROSS_SESSION_LANGUAGE` selects it (`en` by default, or `es`) together with the language the LLM writes
in. Tool errors, logs, search results and the CLI are always in English.
"""

import os

LANGUAGES = ("en", "es")
DEFAULT_LANGUAGE = "en"


def language() -> str:
    value = os.environ.get("CROSS_SESSION_LANGUAGE", DEFAULT_LANGUAGE).strip().lower()[:2]
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


_TEXT = {
    "en": {
        "llm_language": "Write every text field in English, whatever the language of the transcript.",
        "claimed": "(claimed, not verified)",
        "hypothesis": "hypothesis",
        "handoff_title": "Extended Handoff",
        "untitled": "untitled",
        "open_question": "Question left open when the session ended, unanswered: “{question}”",
        "offline_notice": ("> Generated without a language model (the Gemini API failed); not saved. "
                           "For more detail use `get_session_turns` with `query`."),
        "offline_notice_no_key": ("> Generated without a language model because there is no Gemini credential; not saved. "
                                  "Set `GEMINI_API_KEY`, `~/.ai/.gemini_env`, or OAuth via "
                                  "`gcloud auth application-default-login` so a model summarizes the session "
                                  "and the agent does not have to read the transcript. "
                                  "For more detail use `get_session_turns` with `query`."),
        "offline_title": "Recorded title",
        "offline_focus": "Requested focus",
        "offline_first": "First user instruction",
        "offline_status": "Session status",
        "offline_updated": "Last update",
        "offline_turns": "Turns",
        "offline_unknown": "unknown",
        "offline_last_assistant": "Last assistant message",
        "offline_recent": "Recent user instructions",
        "offline_tools": "Latest tool actions",
        "offline_no_llm": "(not available without an LLM)",
        "offline_review": "Review the last user instruction and the last assistant message.",
        "offline_step1": "Move to `{workspace}` and check `git status` and the pending changes.",
        "offline_step2": "Confirm with the user what is still pending from the last instruction.",
        "offline_step3": "Continue with the task: **{title}**.",
        "offline_no_data": "(no data)",
        "resume_title": "SESSION RESUME",
        "resume_workdir": "Original working directory",
        "resume_footer": ("Before acting, check the real state of the repository (`git status`, tests): it may have "
                          "changed after the session.\nThen confirm the context and lay out the next steps you will "
                          "take."),
        "memory_project": "Project",
        "memory_updated": "Updated",
        "memory_model": "Model",
        "memory_unknown": "unknown",
        "memory_session": "session",
        "memory_sessions": "sessions",
        "memory_evidence": "evidence",
        "memory_empty": "_No entries._",
        "memory_no_summary": "(no summary)",
        "section_state": "Current state",
        "section_D": "Active decisions",
        "section_T": "Pitfalls → fix",
        "section_C": "Commands that work",
        "section_P": "Pending",
        "section_A": "AGENTS.md candidates",
        "section_log": "Log",
    },
    "es": {
        "llm_language": ("Write every text field in formal Spanish from Spain (usted, never voseo), whatever the "
                         "language of the transcript."),
        "claimed": "(afirmado, no verificado)",
        "hypothesis": "hipótesis",
        "handoff_title": "Handoff Ampliado",
        "untitled": "sin título",
        "open_question": "Pregunta abierta al cierre de la sesión, sin respuesta: «{question}»",
        "offline_notice": ("> Generado sin modelo de lenguaje (la API de Gemini falló); no se guarda. "
                           "Para más detalle use `get_session_turns` con `query`."),
        "offline_notice_no_key": ("> Generado sin modelo de lenguaje porque no hay credencial de Gemini; no se guarda. "
                                  "Configure `GEMINI_API_KEY`, `~/.ai/.gemini_env` u OAuth con "
                                  "`gcloud auth application-default-login` para que un modelo resuma la "
                                  "sesión y el agente no tenga que leer la transcripción. "
                                  "Para más detalle use `get_session_turns` con `query`."),
        "offline_title": "Título registrado",
        "offline_focus": "Foco solicitado",
        "offline_first": "Primera instrucción del usuario",
        "offline_status": "Estado de la sesión",
        "offline_updated": "Última actualización",
        "offline_turns": "Turnos",
        "offline_unknown": "desconocida",
        "offline_last_assistant": "Último mensaje del asistente",
        "offline_recent": "Instrucciones recientes del usuario",
        "offline_tools": "Últimas acciones de herramientas",
        "offline_no_llm": "(no disponible sin LLM)",
        "offline_review": "Revise la última instrucción del usuario y el último mensaje del asistente.",
        "offline_step1": "Situarse en `{workspace}` y revisar `git status` y los cambios pendientes.",
        "offline_step2": "Confirmar con el usuario qué quedó pendiente de la última instrucción.",
        "offline_step3": "Continuar con la tarea: **{title}**.",
        "offline_no_data": "(sin datos)",
        "resume_title": "RETOMA DE SESIÓN",
        "resume_workdir": "Directorio de trabajo original",
        "resume_footer": ("Antes de actuar, verifique el estado real del repositorio (`git status`, tests): pudo "
                          "cambiar tras la sesión.\nDespués, confirme el contexto y presente los próximos pasos que "
                          "ejecutará."),
        "memory_project": "Proyecto",
        "memory_updated": "Actualizado",
        "memory_model": "Modelo",
        "memory_unknown": "desconocido",
        "memory_session": "sesión",
        "memory_sessions": "sesiones",
        "memory_evidence": "evidencia",
        "memory_empty": "_Sin entradas._",
        "memory_no_summary": "(sin resumen)",
        "section_state": "Estado actual",
        "section_D": "Decisiones vigentes",
        "section_T": "Trampas → solución",
        "section_C": "Comandos que funcionan",
        "section_P": "Pendiente",
        "section_A": "Candidatas a AGENTS.md",
        "section_log": "Bitácora",
    },
}

# Section titles of the Extended Handoff. Only the number ("### N.") is structural; the title is rewritten
# to the canonical one of the current language.
_HANDOFF_SECTIONS = {
    "en": (
        ("Intent and scope", "what the user asked for and why."),
        ("Current state and checks", "what works; which tests or commands passed or failed."),
        ("Decisions and constraints", "each one with its reason (including the user's conventions)."),
        ("Files, commands and artifacts", "paths touched and the outcome of the relevant commands."),
        ("Errors → fix", ""),
        ("Pending and blockers", ""),
        ("Immediate plan", ""),
        ("Suggested skills", "skills or tools that would help to resume."),
    ),
    "es": (
        ("Intención y alcance", "what the user asked for and why."),
        ("Estado actual y verificaciones", "what works; which tests or commands passed or failed."),
        ("Decisiones y restricciones", "each one with its reason (including the user's conventions)."),
        ("Ficheros, comandos y artefactos", "paths touched and the outcome of the relevant commands."),
        ("Errores → solución", ""),
        ("Pendientes y bloqueos", ""),
        ("Plan inmediato", ""),
        ("Skills sugeridas", "skills or tools that would help to resume."),
    ),
}


def t(key: str, **kwargs: str) -> str:
    text = _TEXT[language()][key]
    return text.format(**kwargs) if kwargs else text


def handoff_sections() -> tuple[str, ...]:
    """The eight canonical `### N. Title` headings in the current language."""
    return tuple(f"### {n}. {title}" for n, (title, _) in enumerate(_HANDOFF_SECTIONS[language()], 1))


def handoff_spec() -> str:
    """Section-by-section instructions for the LLM, with the headings in the current language."""
    s = handoff_sections()
    desc = [d for _, d in _HANDOFF_SECTIONS[language()]]
    return f"""EXACTLY these sections, in this order (no preceding header; the system adds it):
  {s[0]} — {desc[0]}
  {s[1]} — {desc[1]}
  {s[2]} — {desc[2]}
  {s[3]} — {desc[3]}
  {s[4]} — paying special attention to the user's corrections. Title each error by its ROOT CAUSE,
      not by the symptom; if the cause was not confirmed in the transcript, mark it as "{t('hypothesis')}".
  {s[5]} — list as pending every user request without evidence of completion
      (a command or tool output that confirms it). Never write "no pending items/blockers" if the last
      user instruction was left unfulfilled.
  {s[6]} — verbatim quote (in `>`) of the "Last user instruction" given in the input, followed by
      a numbered, actionable list with the exact commands to continue without asking again what is settled.
  {s[7]} — {desc[7]}
  Label as "{t('claimed')}" whatever only the assistant stated without command output to back it.
  Reference existing artifacts by path or URL instead of reproducing their content.
  At most ~8,000 characters."""
