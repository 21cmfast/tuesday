"""Configuration for docs."""

import os
from datetime import datetime, timezone

from tuesday import __version__

from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))


class Mock(MagicMock):
    """Make a Mock so that a package doesn't have to actually exist."""

    @classmethod
    def __getattr__(cls, name):
        """Get stuff."""
        return MagicMock()


MOCK_MODULES = [
    "py21cmfast.c_21cmfast",
    "click",
    "tqdm",
    "pyyaml",
    "h5py",
    "cached_property",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
]

if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"

autosectionlabel_prefix_document = True

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = ".rst"
master_doc = "index"
project = "tuesday"
year = str(datetime.now(tz=timezone.utc).year)
author = "21cmFAST Team"
copyright = f"{year}, {author}"
version = release = __version__
templates_path = ["templates"]

pygments_style = "trac"
extlinks = {
    "issue": ("https://github.com/21cmfast/tuesday/issues/%s", "#"),
    "pr": ("https://github.com/21cmfast/tuesday/pull/%s", "PR #"),
}

html_theme = "furo"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "templates",
    "**.ipynb_checkpoints",
]
