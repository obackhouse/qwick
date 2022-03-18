GCC=g++
CFLAGS=-Wall -O2 -std=c++17 -shared -fPIC
SRC=$(shell pwd)/src
BUILD=$(shell pwd)/build
API=$(shell pwd)/api
INCLUDE=-I $(SRC)
PBFLAGS:=$(shell /home/olli/psi4conda/envs/py37/bin/python3.7 -m pybind11 --includes)
PBSUFFIX:=$(shell /home/olli/psi4conda/envs/py37/bin/python3.7-config --extension-suffix)

all: log index.so operator.so expression.so wick.so

log:
	mkdir -p $(BUILD)
	@echo "include:          $(INCLUDE)"
	@echo "pybind11 flags:   $(PBFLAGS)"
	@echo "pybind11 suffix:  $(PBSUFFIX)"

index.so:
	$(GCC) $(INCLUDE) $(PBFLAGS) $(CFLAGS) \
		$(SRC)/util.cc \
		$(SRC)/index.cc \
		$(API)/_index.cc \
		-o $(BUILD)/_index$(PBSUFFIX)

operator.so:
	$(GCC) $(INCLUDE) $(PBFLAGS) $(CFLAGS) \
		$(SRC)/util.cc \
		$(SRC)/index.cc $(SRC)/operator.cc \
		$(API)/_operator.cc \
		-o $(BUILD)/_operator$(PBSUFFIX)

expression.so:
	$(GCC) $(INCLUDE) $(PBFLAGS) $(CFLAGS) \
		$(SRC)/util.cc \
		$(SRC)/index.cc $(SRC)/operator.cc $(SRC)/expression.cc \
		$(API)/_expression.cc \
		-o $(BUILD)/_expression$(PBSUFFIX)

wick.so:
	$(GCC) $(INCLUDE) $(PBFLAGS) $(CFLAGS) \
		$(SRC)/util.cc \
		$(SRC)/index.cc $(SRC)/operator.cc $(SRC)/expression.cc $(SRC)/wick.cc \
		$(API)/_wick.cc \
		-o $(BUILD)/_wick$(PBSUFFIX)

clean:
	rm -f $(BUILD)/index$(PBSUFFIX)
	rm -f $(BUILD)/operator$(PBSUFFIX)
	rm -f $(BUILD)/expression$(PBSUFFIX)
	rm -f $(BUILD)/wick$(PBSUFFIX)
