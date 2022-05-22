build_wheel: clean_build
	python setup.py sdist bdist_wheel
	auditwheel repair dist/*.whl 
	rm -rf dist/*.whl 
	mv wheelhouse/* dist/
	rm -rf wheelhouse
	
clean_build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find ./ -name .dgl | xargs rm -rf
	find ./ -name __pycache__ | xargs rm -rf
	find ./ -name *.c | xargs rm -rf
	find ./ -name *.pyd | xargs rm -rf
	find ./ -name *.so | xargs rm -rf

upload_test: build_wheel
	python3 -m twine upload --repository testpypi dist/*

upload: build_wheel
	python3 -m twine upload dist/*