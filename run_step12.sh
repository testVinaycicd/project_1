mkdir -p graphify-out
echo ".venv/bin/python" > graphify-out/.graphify_python
.venv/bin/python -c "
import json
from graphify.detect import detect
from pathlib import Path
result = detect(Path('.'))
print(json.dumps(result))
" > graphify-out/.graphify_detect.json
