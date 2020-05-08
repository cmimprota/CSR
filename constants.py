import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(ROOT_DIR, "twitter-datasets")

# each preprocessing method gets its own `vocabularies/<method>/full` and `/cut`
# VOCABULARIES_FULL_PATH = os.path.join(ROOT_DIR, "vocabularies", "full")
# VOCABULARIES_CUT_PATH = os.path.join(ROOT_DIR, "vocabularies", "cut")
VOCABULARIES_PATH = os.path.join(ROOT_DIR, "vocabularies")
