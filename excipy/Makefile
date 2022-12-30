
# Uncomment (and edit) this for advanced compiling
# advopt = --build-dir ./tmp --fcompiler=intelem

####################################################

#Find python version
python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python -V 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})
python_version_patch := $(word 3,${python_version_full})
pylib = python${python_version_major}.${python_version_minor}


.PHONY: all clean 

all: tmu.so

ifdef advopt
%.so: %.f90
	# Advanced compiling
	f2py  $(advopt) --opt='-O3'  -c -m $(<:.f90=) $<
	@echo ... correct wrong f2py linking:
	ifort -shared -nofor_main tmp/tmp/src.*/tmumodule.o tmp/tmp/src.*/fortranobject.o tmp/tmu.o -L/usr/lib64 -l${pylib} -o $(<:.f90=).so 	rm -rf tmp
else
%.so: %.f90
	f2py3  --opt='-O3' -lgomp --f90flags='-fopenmp'  -c -m $(<:.f90=) $<
endif

clean:
	rm -rf tmu.so

