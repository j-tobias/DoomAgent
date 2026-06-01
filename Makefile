PYTHON     := python3.13
VENV       := .venv
SITE_PKG   := $(VENV)/lib/$(PYTHON)/site-packages
PTH_FILE   := $(SITE_PKG)/jku_wad.pth

.PHONY: setup
setup: $(VENV) $(PTH_FILE)
	@echo "Ready. Activate with: source $(VENV)/bin/activate"

$(VENV): pyproject.toml uv.lock
	uv sync
	@touch $(VENV)

$(PTH_FILE): $(VENV)
	@echo "$(CURDIR)/jku.wad" > $(PTH_FILE)
