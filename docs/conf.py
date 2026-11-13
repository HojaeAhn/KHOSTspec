"""Build the pipeline's guides and API reference from the checked-out source."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "KHOSTspec"
author = ""
copyright = ""
language = "en"
needs_sphinx = "8.2"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build", "__pycache__"]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

html_theme = "alabaster"
html_title = "KHOSTspec documentation"
html_short_title = "KHOSTspec"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_copyright = False
html_theme_options = {
    "description": "Usage, scientific conventions, and API reference",
    "fixed_sidebar": True,
    "page_width": "1180px",
    "sidebar_width": "260px",
}
html_sidebars = {"**": ["about.html", "globaltoc.html", "searchbox.html"]}

# Autodoc imports the real runtime; missing dependencies must fail the build.
autodoc_mock_imports = []
