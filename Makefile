default:
	python3 test_coefficients.py
	python3 run_benchmark.py
	python3 make_figures.py
	python3 generate_osiris_deck.py
	
install:
	pip install osiris_utils numpy matplotlib --break-system-packages
