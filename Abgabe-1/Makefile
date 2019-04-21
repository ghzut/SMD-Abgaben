all: finish.pdf

finish.pdf: build/FehlerF.pdf finish.tex| build
	TEXINPUTS=build: lualatex --output-directory=build finish.tex
	TEXINPUTS=build: lualatex --output-directory=build finish.tex

build/FehlerF.pdf: Abgabe1.py | build
	TEXINPUTS=$$(pwd): python Abgabe1.py

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
