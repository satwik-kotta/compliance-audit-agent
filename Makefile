.PHONY: ui api test eval
ui:   ; cd frontend && npm install && npm run build
api:  ; uvicorn backend.app.main:app --reload --port 8000
test: ; python -W ignore -m pytest -q tests
eval: ; python -m eval.run_eval
