PD=pandoc

all: presentation.html

presentation.html: presentation.md
	# $(PD) -t revealjs --katex --embed-resources --standalone --slide-level=3 presentation.md -o presentation.html
	$(PD) -t revealjs --katex --standalone --slide-level=3 presentation.md -o presentation.html
