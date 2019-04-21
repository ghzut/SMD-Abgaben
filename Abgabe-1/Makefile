ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
	# Unix
	pwd := $$(pwd)
	translate = $1
else
	# WindowsmitMSYS2/Cygwin
	pwd := $$(cygpath -m "$$(pwd)")
	translate = $(shell echo '$1'| sed 's/:/;/g')
endif

all: finish.pdf

finish.pdf: build/FehlerF.pdf finish.tex| build
	TEXINPUTS=build: lualatex --output-directory=build finish.tex
	TEXINPUTS=build: lualatex --output-directory=build finish.tex

build/FehlerF.pdf: Abgabe1.py | build
	TEXINPUTS="$(call translate,$(pwd):)" python Abgabe1.py

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
