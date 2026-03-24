DOCTYPE = RTN
DOCNUMBER = 095
DOCNAME = $(DOCTYPE)-$(DOCNUMBER)
FLATDIR = forAAS
SUBDIRS = figures

tex = $(filter-out $(wildcard *aglossary.tex) , $(wildcard *.tex))

GITVERSION := $(shell git log -1 --date=short --pretty=%h)
GITDATE := $(shell git log -1 --date=short --pretty=%ad)
GITSTATUS := $(shell git status --porcelain)
ifneq "$(GITSTATUS)" ""
	GITDIRTY = -dirty
endif

export TEXMFHOME ?= lsst-texmf/texmf

SCRIPTS_DIR=scripts
PYTHON_SCRIPTS=$(wildcard $(SCRIPTS_DIR)/*.py)

$(DOCNAME).pdf: $(scripts) $(tex) local.bib authors.tex
	# mv $(DOCNAME).tex orig.tex
	# sed s/twocolumn,//1 orig.tex > $(DOCNAME).tex
	latexmk -bibtex -xelatex -f $(DOCNAME)
	# makeglossaries $(DOCNAME)
	# mv orig.tex $(DOCNAME).tex
	latexmk -bibtex -xelatex -f $(DOCNAME)
	# dont like it much but this removes the error
	echo "\@istfilename{RTN-095.ist}" >> RTN-095.aux
	xelatex $(DOCNAME)

authors.tex:  authors.yaml
	python3 $(TEXMFHOME)/../bin/db2authors.py -m aas7 > authors.tex

authors.txt:  authors.yaml
	python3 $(TEXMFHOME)/../bin/db2authors.py -m arxiv > authors.txt

authors.csv: authors.yaml
	python3 $(TEXMFHOME)/../bin/db2authors.py -m aascsv > authors.csv


aglossary.tex :$(tex) myacronyms.txt
	python3 $(TEXMFHOME)/../bin/generateAcronyms.py -t"Sci DM Gen" -g $(tex)
	
flat:
	if [ ! -d $(FLATDIR) ]; then \
		mkdir $(FLATDIR) ; \
	fi
	latexpand --keep-comments -o $(FLATDIR)/$(DOCNAME).tex $(DOCNAME).tex
	@for dir in $(SUBDIRS); do \
		if [ -d "$$dir" ] && [ -n "$$(ls -A $$dir 2>/dev/null)" ]; then \
			cp $$dir/* $(FLATDIR); \
			echo "  ✓ Copied $$dir"; \
		fi; \
	done
	cp aas*.* $(FLATDIR)
	cp *.bib $(FLATDIR)
	cd $(FLATDIR) &&\
	latexmk -bibtex -xelatex -f $(DOCNAME) &&\
	makeglossaries $(DOCNAME) &&\
	latexmk -bibtex -xelatex -f $(DOCNAME) &&\
	latexmk -c &&\
	rm -f *.gls *.xdv *.glg *.glo *.ist *.bib &&\
	if [ -f README.txt ]; then rm README.txt; fi && \
	echo "Flat files in $(FLATDIR)."

aglossary.tex :$(tex) myacronyms.txt
	python3 $(TEXMFHOME)/../bin/generateAcronyms.py -t"Sci DM Gen" -g $(tex)

.PHONY: clean
clean:
	latexmk -c
	rm -f $(DOCNAME).bbl
	rm -f $(DOCNAME).pdf
	rm -f meta.tex
	rm -f authors.tex
	rm -f $(FLATDIR)/*

.FORCE:

deps:
	pip install -r lsst-texmf/requirements.txt 

authors.yaml:
	python3 $(TEXMFHOME)/../bin/makeAuthorListsFromGoogle.py --builder -p 1yMRqNdPVoAtjBMEPve2WEyt3V_73o4uIv-ZuHvzpeJM "A2:L1000"

skip: .FORCE
	python3 $(TEXMFHOME)/../bin/makeAuthorListsFromGoogle.py --skip `cat skip.count` --builder -p 1yMRqNdPVoAtjBMEPve2WEyt3V_73o4uIv-ZuHvzpeJM "A2:L1000"
	
merge: new_authors.yaml
	python3 $(TEXMFHOME)/../bin/makeAuthorListsFromGoogle.py -m new_authors.yaml
	cp skip skip.count

merge_affil: new_affiliations.yaml
	python3 $(TEXMFHOME)/../bin/makeAuthorListsFromGoogle.py -a new_affiliations.yaml

scripts:
	@echo "Running Python scripts..."
	@for script in $(PYTHON_SCRIPTS); do \
		python3 $$script; \
	done
