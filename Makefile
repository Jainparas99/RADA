.PHONY: setup test lint seed evolve demo

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

seed:
	uv run ratchet init && uv run ratchet seed

evolve:
	uv run ratchet evolve --gens 8

demo:
	uv run ratchet history && uv run ratchet attack && uv run ratchet chart
