@echo off
setlocal
set "DIR=%~dp0.."
if not defined PYTHONPATH (
  set "PYTHONPATH=%DIR%"
) else (
  set "PYTHONPATH=%DIR%;%PYTHONPATH%"
)
set "PYTHONUTF8=1"
if defined CROSS_SESSION_PYTHON (
  "%CROSS_SESSION_PYTHON%" -m cross_session_mcp.cli %*
) else (
  python -m cross_session_mcp.cli %*
)

