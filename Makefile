# elmo · short verbs for the dev loop.
# all targets delegate to the ./up and ./down scripts so behavior stays
# consistent whether you type `./up` or `make up`.

.PHONY: up down restart status logs clean test help

help:
	@echo "  elmo  ::  make up        bootstrap + start (http://127.0.0.1:7777)"
	@echo "  elmo  ::  make down      stop the daemon"
	@echo "  elmo  ::  make restart   down then up"
	@echo "  elmo  ::  make status    is it running?"
	@echo "  elmo  ::  make logs      tail .elmo/daemon.log"
	@echo "  elmo  ::  make test      run the test suite"
	@echo "  elmo  ::  make clean     stop, remove .venv and runtime state"

up:
	@./up

down:
	@./down

restart: down up

status:
	@if [ -f .elmo/daemon.pid ] && kill -0 "$$(cat .elmo/daemon.pid)" 2>/dev/null; then \
	  printf "  elmo  ::  running (pid %s) at http://127.0.0.1:%s\n" "$$(cat .elmo/daemon.pid)" "$${ELMO_PORT:-7777}"; \
	else \
	  printf "  elmo  ::  not running.\n"; \
	fi

logs:
	@tail -f .elmo/daemon.log

test:
	@.venv/bin/pytest -q || ./up >/dev/null 2>&1 && .venv/bin/pytest -q

clean: down
	@rm -rf .venv .elmo
	@printf "  elmo  ::  cleaned .venv and .elmo\n"
