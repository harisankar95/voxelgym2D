# Generate HTML documentation with Sphinx
docs-html:
	sphinx-apidoc -f -e -M -o docs/ voxelgym2D/
	sphinx-build -b html docs docs/_build