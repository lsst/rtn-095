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
	python3 $(TEXMFHOME)/../bin/db2authors.py -m aas7 > authors.tex

authors.txt:  authors.txt
	python3 $(TEXMFHOME)/../bin/db2authors.py -m arxiv > authors.txt

aglossary.tex :$(tex) myacronyms.txt
	python3 $(TEXMFHOME)/../bin/generateAcronyms.py -t"Sci DM" -g $(tex)


.PHONY: clean
clean:
	latexmk -c
	rm -f $(DOCNAME).{bbl,glsdefs,pdf}
	rm -f authors.tex

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
