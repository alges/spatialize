all: setup.py
	python3 setup.py build_ext --inplace

clean:
	rm -Rf build* *.so


test: all
	DYLD_LIBRARY_PATH=$${DYLD_LIBRARY_PATH:-/opt/homebrew/opt/libomp/lib} PYTHONPATH=.:src/python \
		sh -c 'python3 examples/scripted_examples/esi_3d_nongriddata.py && \
		       python3 -m pytest -q tests/refactor_guard && \
		       python3 -m spatialize.scenarios'
