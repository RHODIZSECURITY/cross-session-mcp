@echo off
setlocal
set "DIR=%~dp0.."
if not defined PYTHONPATH (
  set "PYTHONPATH=%DIR%"
) else (
  set "PYTHONPATH=%DIR%;%PYTHONPATH%"
)
set "PYTHONUTF8=1"
if not defined CROSS_SESSION_OPENAI_BASE (
  set "CROSS_SESSION_OPENAI_BASE=http://localhost:10102/v1"
)
if not defined CROSS_SESSION_OPENAI_KEY (
  set "CROSS_SESSION_OPENAI_KEY=ocx_data_5363e86ca471a28312de9fd918b7f4b53aba1565"
)
if not defined CROSS_SESSION_MODELS (
  set "CROSS_SESSION_MODELS=google-antigravity/gemini-3.8-flash,google/gemini-3.5-flash-lite"
)
if defined CROSS_SESSION_PYTHON (
  "%CROSS_SESSION_PYTHON%" -m cross_session_mcp.cli %*
) else (
  python -m cross_session_mcp.cli %*
)

