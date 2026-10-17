.PHONY: run-frontend-local run-backend-local run-local

run-backend-local:
	cd backend && uv run main

run-frontend-local:
	cd frontend && npm run dev

run-local:
	$(MAKE) -j2 run-backend-local run-frontend-local
