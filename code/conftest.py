# conftest.py — tells pytest that this directory is the import root.
# Without this file, pytest tries to import test_classifier as
# 'code.test_classifier' (treating the parent folder as a package),
# which fails because 'code' is not an installable package.
