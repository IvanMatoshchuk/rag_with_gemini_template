import os
import tempfile

os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="tech-test-")
os.environ["GOOGLE_SEARCH_ENABLED"] = "true"
