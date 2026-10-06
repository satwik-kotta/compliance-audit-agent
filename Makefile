.PHONY: ui api test eval pages
ui:   ; cd frontend && npm install && npm run build
api:  ; uvicorn backend.app.main:app --reload --port 8000
test: ; python -W ignore -m pytest -q tests
eval: ; python -m eval.run_eval
pages: ; python -m backend.app.export_static && cd frontend && VITE_STATIC=1 npx vite build --base ./ --outDir ../docs
