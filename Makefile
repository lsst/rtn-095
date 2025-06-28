DOCTYPE = RTN
DOCNUMBER = 095
DOCNAME = $(DOCTYPE)-$(DOCNUMBER)

tex = $(filter-out $(wildcard *aglossary.tex) , $(wildcard *.tex))

GITVERSION := $(shell git log -1 --date=short --pretty=%h)
GITDATE := $(shell git log -1 --date=short --pretty=%ad)
GITSTATUS := $(shell git status --porcelain)
ifneq "$(GITSTATUS)" ""
	GITDIRTY = -dirty
endif

export TEXMFHOME ?= lsst-texmf/texmf

$(DOCNAME).pdf: $(tex) local.bib authors.tex aglossary.tex
	mv $(DOCNAME).tex orig.tex
	sed s/twocolumn,//1 orig.tex > $(DOCNAME).tex 
	latexmk -bibtex -xelatex -f $(DOCNAME)
	makeglossaries $(DOCNAME)
	mv orig.tex $(DOCNAME).tex
	latexmk -bibtex -xelatex -f $(DOCNAME)
	# dont like it much ut this removes the error 
	echo "\@istfilename{RTN-095.ist}" >> RTN-095.aux
	xelatex $(DOCNAME)

authors.tex:  authors.yaml
	python3 $(TEXMFHOME)/../bin/db2authors.py -m aas > authors.tex

aglossary.tex :$(tex) myacronyms.txt
	generateAcronyms.py -t"Sci DM" -g $(tex)


.PHONY: clean
clean:
	latexmk -c
	rm -f $(DOCNAME).{bbl,glsdefs,pdf}
	rm -f authors.tex

.FORCE:
