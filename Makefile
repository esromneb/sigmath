.PHONY: all clean test testworking testnogui

all:
	@echo "try:"
	@echo "     make test"

clean:
	rm -f *.pyc

test: testworking

testworking:
	scripts/run_working_tests.sh

testnogui:
	UNITTEST_NO_X11=1 scripts/run_working_tests.sh

plot:
	python test_sigmath.py plot