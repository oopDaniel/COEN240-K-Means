remove:
	rm -rf ./tmp

install:
	@echo Creating virtual environment...
	python3 -m venv tmp
	@echo Installing pkg...
	./tmp/bin/pip3 install xlrd numpy matplotlib

# modify the arguments here
start:
	./tmp/bin/python3 main.py Iris.xls

all:
	make remove install start

.PHONY: remove install start all