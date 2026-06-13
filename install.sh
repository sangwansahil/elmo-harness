#!/usr/bin/env sh
# elmo install — pip-based, picks the right hardware extra.
set -eu

REPO="git+https://github.com/sangwansahil/elmo-harness"
EXTRA="mlx"

# Pick CUDA if nvidia-smi exists.
if command -v nvidia-smi >/dev/null 2>&1; then
  EXTRA="cuda"
fi

# Honor an explicit override: `curl ... | EXTRA=ui sh`
EXTRA="${ELMO_EXTRA:-$EXTRA}"

echo "  elmo  ::  installing with [$EXTRA] extra"
if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -w "$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo /usr/lib)" ]; then
  echo "  elmo  ::  no venv detected — adding --user"
  python3 -m pip install --user "elmo-harness[$EXTRA] @ $REPO"
else
  python3 -m pip install "elmo-harness[$EXTRA] @ $REPO"
fi

echo
echo "  elmo  ::  installed. next:"
echo "      elmo init"
echo "      elmo run examples/function-calling.yaml"
echo
