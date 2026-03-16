# Project Rules

## Reference Check
Before assuming any technical details or asking the user for project-specific information, the agent MUST:
1. Consult [docs/references.md](file:///home/df/data/jflinte/SkyScenes/docs/references.md).
2. If necessary, follow the links in `docs/references.md` to external resources (Paper, HuggingFace, GitHub, etc.) to get the most accurate and up-to-date information.
3. Check [docs/README.md](file:///home/df/data/jflinte/SkyScenes/docs/README.md) for local setup and overview instructions.

## Storage
- All generated datasets (images, metadata, segmentation maps, etc.) MUST be stored in `/home/df/data/datasets`.
- Ensure scripts use this path as the default root directory for data output.

## Package Management
- Use `uv` as the package manager for all Python-related tasks.
- Always run scripts using `uv run <script_path>`.
- Manage dependencies via `uv add` and `uv remove`.

This ensures all work aligns with the official SkyScenes dataset specifications and research methodology.
