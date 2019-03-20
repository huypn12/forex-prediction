all: report

clean:
	rm *.{aux,bbl,blg,dvi,pdf,log}

pdf: report.dvi
	dvipdfm report.dvi

report: report.tex report.bib
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex


