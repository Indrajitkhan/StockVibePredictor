# StockVibePredictor — ui/Makefile (React frontend, CRA + CRACO)
#
# Frontend commands. Adopted from the SHADE-ai / VisionCore Makefile architecture (Task 07).

.DEFAULT_GOAL := help

SHELL := /bin/sh

.PHONY: help install dev start test test-watch test-coverage build doctor status stop audit clean

NPM ?= npm
# Informational only: craco.config.js pins the dev server to port 3000
# (known limitation — see docs/CRITICAL-ISSUES.md).
PORT ?= 3000

RESET := \033[0m
BOLD := \033[1m
DIM := \033[2m
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
MAGENTA := \033[35m

help:
	@printf '\n$(MAGENTA)StockVibePredictor — ui (React frontend, CRA + CRACO)$(RESET)\n\n'
	@printf '$(BOLD)Setup$(RESET)\n'
	@printf '  $(GREEN)make install$(RESET)           Install npm dependencies\n'
	@printf '$(BOLD)Run$(RESET)\n'
	@printf '  $(CYAN)make start$(RESET) / $(CYAN)make dev$(RESET)   Start the CRA dev server on port $(PORT) (opens the browser)\n'
	@printf '$(BOLD)Quality$(RESET)\n'
	@printf '  $(CYAN)make test$(RESET)             Run the Jest suite once (craco test --watchAll=false)\n'
	@printf '  $(CYAN)make test-watch$(RESET)       Run Jest in watch mode\n'
	@printf '  $(CYAN)make test-coverage$(RESET)   Run Jest with coverage\n'
	@printf '  $(CYAN)make build$(RESET)            Create the production bundle in build/\n'
	@printf '  $(CYAN)make audit$(RESET)            Run npm security audit\n'
	@printf '$(BOLD)Maintain$(RESET)\n'
	@printf '  $(CYAN)make doctor$(RESET) / $(CYAN)make status$(RESET)   Show environment readiness\n'
	@printf '  $(CYAN)make stop$(RESET)            Stop the dev server on port $(PORT)\n'
	@printf '  $(CYAN)make clean$(RESET)           Remove node_modules/ and build output (retry-verified)\n\n'
	@printf '$(DIM)No lint/format targets: package.json defines no lint/format scripts (adding dependencies is out of scope).$(RESET)\n'
	@printf '$(DIM)Port note: craco.config.js pins the dev server to $(PORT) (known limitation, docs/CRITICAL-ISSUES.md).$(RESET)\n\n'

install:
	@printf '$(MAGENTA)Installing frontend dependencies...$(RESET)\n'
	@$(NPM) install --no-audit --no-fund
	@printf '$(GREEN)✔ Frontend dependencies installed.$(RESET)\n'

dev start:
	@test -d node_modules || { printf '$(RED)✘ node_modules is missing — run make install first.$(RESET)\n'; exit 1; }
	@printf '\n$(MAGENTA)Starting the CRA development server...$(RESET)\n'
	@printf 'Frontend: $(MAGENTA)http://localhost:$(PORT)$(RESET)\n'
	@printf '$(DIM)The dev server opens the browser automatically; use BROWSER=none make start to suppress it.$(RESET)\n\n'
	@$(NPM) start

test:
	@printf '$(MAGENTA)Running the Jest suite...$(RESET)\n'
	@$(NPM) test
	@printf '$(GREEN)✔ Frontend tests completed.$(RESET)\n'

test-watch:
	@$(NPM) run test:watch

test-coverage:
	@$(NPM) run test:coverage

build:
	@printf '$(MAGENTA)Building the production bundle...$(RESET)\n'
	@$(NPM) run build
	@test -d build || { printf '$(RED)✘ build/ was not produced.$(RESET)\n'; exit 1; }
	@printf '$(GREEN)✔ Production bundle ready in ui/build/.$(RESET)\n'

doctor:
	@printf '  Node:         '; command -v node >/dev/null 2>&1 && printf '$(GREEN)✔ %s$(RESET)\n' "$$(node --version)" || printf '$(RED)✘ missing$(RESET)\n'
	@printf '  npm:          '; command -v npm >/dev/null 2>&1 && printf '$(GREEN)✔ %s$(RESET)\n' "$$(npm --version)" || printf '$(RED)✘ missing$(RESET)\n'
	@printf '  node_modules: '; test -d node_modules && printf '$(GREEN)✔ ready$(RESET)\n' || printf '$(YELLOW)○ missing — run make install$(RESET)\n'
	@printf '  lockfile:     '; test -f package-lock.json && printf '$(GREEN)✔ present$(RESET)\n' || printf '$(YELLOW)○ missing$(RESET)\n'
	@printf '  build:        '; test -d build && printf '$(GREEN)✔ present$(RESET)\n' || printf '$(DIM)○ not created$(RESET)\n'

status: doctor

stop:
	@printf '$(RED)Stopping the frontend dev server on port $(PORT)...$(RESET)\n'
	@if command -v lsof >/dev/null 2>&1; then \
		for pid in $$(lsof -ti:$(PORT) 2>/dev/null); do kill $$pid 2>/dev/null || true; done; \
		sleep 1; \
		for pid in $$(lsof -ti:$(PORT) 2>/dev/null); do kill -9 $$pid 2>/dev/null || true; done; \
	fi
	@printf '$(GREEN)✔ Frontend stopped.$(RESET)\n'

audit:
	@$(NPM) audit

# macOS can fail a single recursive rm with "Directory not empty" when Finder/iCloud
# touches a directory while its contents are being removed (verified: contents are gone,
# the top-level rmdir loses the race). Retry the removal and VERIFY the directory is
# actually gone; fail loudly after 5 attempts. package-lock.json and sources are never
# touched, and removal failures are never hidden.
clean:
	@printf '$(MAGENTA)Cleaning frontend build output and dependencies...$(RESET)\n'
	@for dir in node_modules build dist coverage; do \
		if [ -e $$dir ]; then \
			attempt=0; \
			while [ -e $$dir ]; do \
				attempt=$$((attempt + 1)); \
				if [ $$attempt -gt 5 ]; then \
					printf '$(RED)✘ %s still present after 5 removal attempts.$(RESET)\n' "$$dir"; \
					exit 1; \
				fi; \
				rm -rf $$dir && break || sleep 1; \
			done; \
			if [ -e $$dir ]; then \
				printf '$(RED)✘ %s was not removed.$(RESET)\n' "$$dir"; \
				exit 1; \
			fi; \
		fi; \
	done
	@printf '$(GREEN)✔ Frontend cleaned (node_modules/, build output). package-lock.json preserved.$(RESET)\n'
