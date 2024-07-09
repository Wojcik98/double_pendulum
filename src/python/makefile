default: double_pendulum_python

double_pendulum_python: setup.py
	pip install -e . --config-settings editable_mode=strict

full: setup.py
	pip install -e .[all] --config-settings editable_mode=strict
