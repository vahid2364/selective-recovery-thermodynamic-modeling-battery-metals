#!/bin/bash
set -e
TEXBIN=/Library/TeX/texbin

$TEXBIN/pdflatex -interaction=nonstopmode main5.tex
$TEXBIN/bibtex main5
$TEXBIN/pdflatex -interaction=nonstopmode main5.tex
$TEXBIN/pdflatex -interaction=nonstopmode main5.tex

echo "Done — main5.pdf built."
