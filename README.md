# SSSP delta stepping with charm4py

*Domracheva Daria, 523*

This is an assignment for course "Parallel Large scale Graph Processing".

`main.py` and `utils.py` contain code for SSSP delta-stepping algorithm for distributed memory model. `charm4py` was chosen as a a programming model for this algorithm.

****

## Requirements (python):
* `charm4py`
* `numpy`
* `time`
* `argparse`
* `math`

*Note: `charm4py` requires also `greenlet` and `cython`.*

[charm4py documentation](https://charm4py.readthedocs.io/en/latest)

[installation of charm4py](https://charm4py.readthedocs.io/en/latest/install.html)

## Execution
```bash
python3 -m charmrun.start +pN main.py [options]
```

#### Possible options:
* `-i [FILENAME]`: input file
* `-o [FILENAME]`: output file
* `-r [ROOT]`: index of root vertex
* `-d`: print debug info

If no arguments are provided, prints help information about charm4py and main.py.

## Log and comparison
Run a script which compares outputs of serial (C/C++) and parallel (charm4py) algorithms. Graph sizes: 16 (with debug output), 32, 64 and 128 vertices.
```bash
script.sh > log.txt 2>&1
```
