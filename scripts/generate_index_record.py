"""生成 indexed_files.json 索引记录文件

扫描笔记源目录，记录每个 .md 文件的相对路径和最后修改时间。
用于增量更新时对比哪些文件是新增、修改或已删除。
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone


def main():
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "notes_backup" / "personal_notes"
    output_path = project_root / "chroma_db" / "indexed_files.json"

    if not docs_dir.exists():
        print(f"错误：笔记目录不存在 - {docs_dir}")
        return

    # 确保 chroma_db 目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {}
    for root, dirs, files in os.walk(docs_dir):
        for fn in files:
            if fn.endswith(".md"):
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, docs_dir)
                mtime_timestamp = os.path.getmtime(abs_path)
                # 存为 ISO 格式字符串，方便人工查看和对比
                mtime_str = datetime.fromtimestamp(
                    mtime_timestamp, tz=timezone.utc
                ).isoformat()
                record[rel_path] = mtime_str

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"已生成索引记录: {output_path}")
    print(f"共 {len(record)} 个 .md 文件")


if __name__ == "__main__":
    main()
