import os
from pathlib import Path

EXCLUDE = {".venv", "venv", "__pycache__", ".git", ".idea", "node_modules"}

def should_skip(path):
    return any(part in str(path) for part in EXCLUDE)

def pack_project(root="."):
    output = []
    output.append("=== PROJECT STRUCTURE ===")
    for path in sorted(Path(root).rglob("*")):
        if path.is_dir() and not should_skip(path):
            output.append(f"[DIR]  {path}")
        elif path.is_file() and path.suffix == ".py" and not should_skip(path):
            output.append(f"[FILE] {path}")

    output.append("\n=== PYTHON FILES CONTENT ===")
    for file in sorted(Path(root).rglob("*.py")):
        if not should_skip(file):
            output.append(f"\n===== FILE: {file} =====")
            try:
                output.append(file.read_text(encoding="utf-8"))
            except Exception as e:
                output.append(f"# Ошибка чтения: {e}")

    Path("project_packed.txt").write_text("\n".join(output), encoding="utf-8")
    print("✅ Сохранено в project_packed.txt")

if __name__ == "__main__":
    pack_project()