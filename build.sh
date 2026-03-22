#!/bin/bash
pip install -r requirements.txt
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
