
name := article
dir := Latex
path := $(dir)/$(name)

latex := cd $(dir) && pdflatex -synctex=1 $(name).tex

latex_files :=

figures := $(wildcard Pyfig/fig*.*)

.PHONY: all clean cleanall help

all: $(path).pdf

help:
	@cat README.rst

clean:
	rm -f $(dir)/*.log $(dir)/*.aux $(dir)/*.out $(dir)/*.bbl $(dir)/*.blg $(dir)/*.tmp

cleanall: clean
	rm -rf tmp
	rm -f $(path).pdf

edittex:
	emacs $(path).tex &

startworking: edittex $(path).pdf
	evince $(path).pdf &

vimtex:
	gvim -p $(path).tex $(latex_files) --servername GVIM &
	# NVIM_LISTEN_ADDRESS=GVIM nvim-gtk $(path).tex &

doit: vimtex $(path).pdf
	zathura $(path).pdf &

$(path).pdf: $(path).log $(path).bbl
	@if [ `grep "Package rerunfilecheck Warning: File" $(path).log | wc -l` != 0 ]; then $(latex); fi
	@if [ `grep "Rerun to get cross-references right." $(path).log | wc -l` != 0 ]; then $(latex); fi
	@if [ `grep "Package natbib Warning: Citation(s) may have changed." $(path).log | wc -l` != 0 ]; then $(latex); fi

$(path).log: $(path).tex $(figures) $(latex_files)
	$(latex)

$(path).bbl: $(path).aux $(dir)/biblio.bib
	cd $(dir) && bibtex $(name).aux

pyfig:
	$(foreach make_fig,$(wildcard Python/make_fig*.py),python $(make_fig);)

listfig:
	cd $(dir) && python ../Python/flatex.py $(name).tex /dev/stdout | \
		pandoc -f latex -t rst | \
	       	awk 'BEGIN{RS="\n\n"; print "List of Figures\n===============\n"} /\.\. figure/{print $1; print "\n";}' | \
       		sed '/\.\.\ figure/ s/$$/.png/' > ../Pyfig/README.rst

test:
	FLUID_COMPILE_CACHEDJIT=1 TRANSONIC_COMPILE_JIT=1 TRANSONIC_DEBUG=1 fluidsim-test -m solvers.sw1l.onlywaves
