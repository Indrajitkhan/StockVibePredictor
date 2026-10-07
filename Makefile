# StockVibePredictor — Root Makefile (project orchestrator)
#
# Delegates backend work to api/Makefile and frontend work to ui/Makefile.
# Architecture adopted from the SHADE-ai / VisionCore Makefile system (Task 07).

.DEFAULT_GOAL := help

SHELL := /bin/sh

.PHONY: help install setup dev api-dev ui-dev check test test-api test-ui api-test ui-test build api-regression newman logs status stop clean clean-api clean-ui api-clean ui-clean

RESET := \033[0m
BOLD := \033[1m
DIM := \033[2m
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
MAGENTA := \033[35m

API_DIR := api
UI_DIR := ui

help:
	@printf '\n$(CYAN)StockVibePredictor — project orchestrator$(RESET)\n'
	@printf 'Backend targets live in api/Makefile; frontend targets live in ui/Makefile.\n\n'
	@printf '$(BOLD)Setup$(RESET)\n'
	@printf '  $(GREEN)make install$(RESET)            Install backend (shared root .venv) and frontend dependencies\n'
	@printf '  $(GREEN)make setup$(RESET)              Install dependencies and apply database migrations\n'
	@printf '$(BOLD)Development$(RESET)\n'
	@printf '  $(CYAN)make dev$(RESET)                Start backend (127.0.0.1:8000) and frontend (3000) together\n'
	@printf '  $(CYAN)make api-dev$(RESET)            Start only the Django backend (--noreload; see api/Makefile)\n'
	@printf '  $(CYAN)make ui-dev$(RESET)             Start only the React frontend (CRA, opens the browser)\n'
	@printf '$(BOLD)Quality$(RESET)\n'
	@printf '  $(CYAN)make check$(RESET)              Run Django system checks\n'
	@printf '  $(CYAN)make test$(RESET)               Run backend (Django runner) and frontend (Jest) suites\n'
	@printf '  $(CYAN)make api-test$(RESET)           Run only the backend suite (currently 0 tests)\n'
	@printf '  $(CYAN)make ui-test$(RESET)            Run only the Jest suite\n'
	@printf '  $(CYAN)make build$(RESET)              Build the frontend production bundle (ui/build/)\n'
	@printf '  $(CYAN)make api-regression$(RESET)    Run the Newman API regression suite (backend must be running)\n'
	@printf '$(BOLD)Maintenance$(RESET)\n'
	@printf '  $(CYAN)make logs$(RESET)               Tail the backend log\n'
	@printf '  $(CYAN)make status$(RESET)             Show backend and frontend readiness\n'
	@printf '  $(CYAN)make stop$(RESET)               Stop dev servers on ports 8000 and 3000\n'
	@printf '  $(CYAN)make clean$(RESET)              Remove caches, ui/node_modules/ and build output (SAFE)\n\n'
	@printf '$(DIM)No lint/format targets: no linters are installed in the venv and ui/package.json has no lint scripts.$(RESET)\n'
	@printf '$(DIM)make clean never touches api/.env, api/db.sqlite3, api/app/api/logs/, or model .pkl artifacts.$(RESET)\n\n'

install:
	@printf '$(MAGENTA)Installing backend dependencies (shared root .venv)...$(RESET)\n'
	@$(MAKE) -C $(API_DIR) install
	@printf '$(MAGENTA)Installing frontend dependencies...$(RESET)\n'
	@$(MAKE) -C $(UI_DIR) install
	@printf '$(GREEN)✔ All dependencies installed.$(RESET)\n'

setup: install
	@$(MAKE) -C $(API_DIR) migrate
	@printf '$(GREEN)✔ Project setup completed — run make dev.$(RESET)\n'

dev:
	@printf '\n$(BOLD)$(MAGENTA)Starting StockVibePredictor (backend + frontend)...$(RESET)\n'
	@printf 'Backend:  $(CYAN)http://127.0.0.1:8000$(RESET)\n'
	@printf 'Frontend: $(CYAN)http://localhost:3000$(RESET)\n'
	@printf '$(DIM)Press Ctrl-C to stop both servers.$(RESET)\n\n'
	@$(MAKE) -j 2 api-dev ui-dev

api-dev:
	@$(MAKE) -C $(API_DIR) dev

ui-dev:
	@$(MAKE) -C $(UI_DIR) start

check:
	@$(MAKE) -C $(API_DIR) check

test: test-api test-ui
	@printf '$(GREEN)✔ All tests completed.$(RESET)\n'

test-api api-test:
	@$(MAKE) -C $(API_DIR) test

test-ui ui-test:
	@$(MAKE) -C $(UI_DIR) test

build:
	@$(MAKE) -C $(UI_DIR) build

# Canonical API regression (Newman). The backend must already be running:
#   make api-dev   (in another terminal), then:   make api-regression
api-regression:
	@printf '$(MAGENTA)Running the Newman API regression suite...$(RESET)\n'
	@if ! curl --fail --silent --connect-timeout 2 --max-time 4 http://127.0.0.1:8000/api/system/health/ >/dev/null 2>&1; then \
		printf '$(RED)✘ Backend not reachable on 127.0.0.1:8000 — start it first (make api-dev).$(RESET)\n'; \
		exit 1; \
	fi
	@newman run postman/Collections/StockVibePredictor_API_v2.postman_collection.json \
		-e postman/Environments/Development.postman_environment.json

newman: api-regression

logs:
	@$(MAKE) -C $(API_DIR) logs

status:
	@printf '$(BOLD)Backend$(RESET)\n'
	@$(MAKE) -C $(API_DIR) doctor
	@printf '$(BOLD)Frontend$(RESET)\n'
	@$(MAKE) -C $(UI_DIR) doctor

stop:
	@$(MAKE) -C $(API_DIR) stop
	@$(MAKE) -C $(UI_DIR) stop

# SAFE cleanup: backend caches + ui/node_modules/ and ui build output only.
# Never removes api/.env, api/db.sqlite3, api/app/api/logs/, api/app/api/scripts/
# (Models/Performance/Validation), source, or Git metadata. Delegates to the
# domain Makefiles; removal failures are retried and verified, never suppressed.
clean: clean-api clean-ui
	@printf '$(GREEN)✔ Cleanup completed.$(RESET)\n'

clean-api api-clean:
	@$(MAKE) -C $(API_DIR) clean

clean-ui ui-clean:
	@$(MAKE) -C $(UI_DIR) clean
