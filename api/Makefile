# StockVibePredictor — api/Makefile (Django backend)
#
# Backend commands. Uses the shared root virtualenv (../.venv, Python 3.11).
# Adopted from the SHADE-ai / VisionCore Makefile architecture (Task 07).

.DEFAULT_GOAL := help

SHELL := /bin/sh

.PHONY: help install setup-env dev run check test shell migrate makemigrations migrations logs doctor status stop clean

PYTHON ?= python3.11
VENV ?= ../.venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
MANAGE := $(PY) manage.py
PORT ?= 8000
HOST ?= 127.0.0.1
LOG_FILE := app/api/logs/StockPredict.log

RESET := \033[0m
BOLD := \033[1m
DIM := \033[2m
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
MAGENTA := \033[35m

help:
	@printf '\n$(CYAN)StockVibePredictor — api (Django backend)$(RESET)\n'
	@printf 'Uses the shared root virtualenv ($(VENV), Python 3.11).\n\n'
	@printf '$(BOLD)Setup$(RESET)\n'
	@printf '  $(GREEN)make install$(RESET)        Create/verify the venv and install requirements.txt\n'
	@printf '  $(GREEN)make setup-env$(RESET)     Create api/.env from api/.env.example if missing\n'
	@printf '$(BOLD)Run$(RESET)\n'
	@printf '  $(CYAN)make dev$(RESET)            Start the Django server on $(HOST):$(PORT) (--noreload)\n'
	@printf '  $(CYAN)make run$(RESET)            Alias for dev\n'
	@printf '  $(CYAN)make shell$(RESET)          Open a Django shell\n'
	@printf '$(BOLD)Quality$(RESET)\n'
	@printf '  $(CYAN)make check$(RESET)         Run Django system checks\n'
	@printf '  $(CYAN)make test$(RESET)          Run the Django test suite (currently 0 tests)\n'
	@printf '$(BOLD)Database$(RESET)\n'
	@printf '  $(CYAN)make migrate$(RESET)           Apply migrations\n'
	@printf '  $(CYAN)make makemigrations$(RESET)   Create new migrations\n'
	@printf '  $(CYAN)make migrations$(RESET)        Show migration status\n'
	@printf '$(BOLD)Maintain$(RESET)\n'
	@printf '  $(CYAN)make logs$(RESET)          Tail the backend log ($(LOG_FILE))\n'
	@printf '  $(CYAN)make doctor$(RESET) / status   Show environment readiness\n'
	@printf '  $(CYAN)make clean$(RESET)         Remove Python caches only (db, .env, logs, models untouched)\n\n'
	@printf '$(DIM)No training target on purpose: the ML pipeline has known integrity issues (docs/CRITICAL-ISSUES.md).$(RESET)\n\n'

# The shared virtualenv lives at the repository root; create it there if missing.
$(PY):
	@printf '$(MAGENTA)Creating the shared virtualenv at the repository root ($(VENV))...$(RESET)\n'
	@command -v $(PYTHON) >/dev/null 2>&1 || { printf '$(RED)✘ Python not found: $(PYTHON)$(RESET)\n'; exit 1; }
	@cd .. && $(PYTHON) -m venv .venv
	@$(PIP) install --upgrade pip --disable-pip-version-check >/dev/null
	@printf '$(GREEN)✔ Virtualenv ready.$(RESET)\n'

install: $(PY) setup-env
	@printf '$(MAGENTA)Installing backend dependencies...$(RESET)\n'
	@$(PIP) install -r requirements.txt --disable-pip-version-check
	@printf '$(GREEN)✔ Backend dependencies installed.$(RESET)\n'

setup-env:
	@if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; printf '$(GREEN)✔ Created .env from .env.example$(RESET)\n'; else printf '$(DIM)· Environment file already configured.$(RESET)\n'; fi

dev: $(PY)
	@$(PY) -c 'import django' >/dev/null 2>&1 || { printf '$(RED)✘ Django is not installed — run make install first.$(RESET)\n'; exit 1; }
	@printf '\n$(BOLD)$(CYAN)Django backend$(RESET)\n'
	@printf 'Backend: $(CYAN)http://$(HOST):$(PORT)$(RESET)\n'
	@printf 'API:     $(CYAN)http://$(HOST):$(PORT)/api/v1/$(RESET)\n'
	@printf 'Health:  $(CYAN)http://$(HOST):$(PORT)/api/v1/system/health/$(RESET)\n'
	@printf '$(DIM)--noreload is required until the import-time SIGTERM handler in app/StockPredict/views.py is decoupled (deferred — docs/CRITICAL-ISSUES.md).$(RESET)\n\n'
	@$(MANAGE) runserver $(HOST):$(PORT) --noreload

run: dev

check: $(PY)
	@printf '$(MAGENTA)Running Django system checks...$(RESET)\n'
	@$(MANAGE) check
	@printf '$(GREEN)✔ Django checks passed.$(RESET)\n'

test: $(PY)
	@printf '$(MAGENTA)Running the Django test suite...$(RESET)\n'
	@# The suite is currently empty (0 tests); scaffolding is tracked in docs/CRITICAL-ISSUES.md.
	@$(MANAGE) test
	@printf '$(GREEN)✔ Backend tests completed.$(RESET)\n'

shell: $(PY)
	@$(MANAGE) shell

migrate: $(PY)
	@printf '$(MAGENTA)Applying migrations...$(RESET)\n'
	@$(MANAGE) migrate
	@printf '$(GREEN)✔ Migrations applied.$(RESET)\n'

makemigrations: $(PY)
	@printf '$(MAGENTA)Creating migrations...$(RESET)\n'
	@$(MANAGE) makemigrations
	@printf '$(GREEN)✔ Migrations created.$(RESET)\n'

migrations: $(PY)
	@$(MANAGE) showmigrations

stop:
	@printf '$(RED)Stopping the backend server on port $(PORT)...$(RESET)\n'
	@if command -v lsof >/dev/null 2>&1; then \
		for pid in $$(lsof -ti:$(PORT) 2>/dev/null); do kill $$pid 2>/dev/null || true; done; \
		sleep 1; \
		for pid in $$(lsof -ti:$(PORT) 2>/dev/null); do kill -9 $$pid 2>/dev/null || true; done; \
	fi
	@printf '$(GREEN)✔ Backend stopped.$(RESET)\n'

logs:
	@if [ -f $(LOG_FILE) ]; then tail -n 80 $(LOG_FILE); else printf '$(YELLOW)No backend log yet ($(LOG_FILE)) — start the server first.$(RESET)\n'; fi

doctor:
	@printf '  Python:  '; test -x $(PY) && printf '$(GREEN)✔ %s$(RESET)\n' "$$($(PY) --version 2>&1)" || printf '$(YELLOW)○ venv missing — run make install$(RESET)\n'
	@printf '  Django:  '; test -x $(PY) && $(PY) -c 'import django' >/dev/null 2>&1 && printf '$(GREEN)✔ installed$(RESET)\n' || printf '$(YELLOW)○ unavailable$(RESET)\n'
	@printf '  .env:    '; test -f .env && printf '$(GREEN)✔ present$(RESET)\n' || printf '$(YELLOW)○ missing — run make setup-env$(RESET)\n'
	@printf '  DB:      '; test -f db.sqlite3 && printf '$(GREEN)✔ db.sqlite3 present$(RESET)\n' || printf '$(YELLOW)○ not created$(RESET)\n'
	@printf '  Log:     '; test -f $(LOG_FILE) && printf '$(GREEN)✔ present$(RESET)\n' || printf '$(DIM)○ not created yet$(RESET)\n'

status: doctor

clean:
	@printf '$(MAGENTA)Removing backend caches...$(RESET)\n'
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type f -name '*.pyc' -delete
	@rm -rf .pytest_cache .coverage htmlcov
	@printf '$(GREEN)✔ Backend caches removed.$(RESET)\n'
	@printf '$(DIM)Untouched: .env, db.sqlite3, app/api/logs/, app/api/scripts/ (models, performance, validation).$(RESET)\n'
