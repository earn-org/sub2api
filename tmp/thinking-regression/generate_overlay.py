"""Print a machine-local Go overlay without copying tests into production paths."""

import json
from pathlib import Path


directory = Path(__file__).resolve().parent
root = directory.parent.parent
print(
    json.dumps(
        {
            "Replace": {
                str(root / "backend/internal/service/artex_thinking_regression_test.go"): str(
                    directory / "thinking_regression_test.go"
                )
            }
        },
        indent=2,
    )
)
