## acaiplus

Add the project description here.

## Setup virtualenv and install dependencies

* Create a python 3 virtual environment
```bash    
virtualenv -p python3 ~/virtualenvs/acaiplus
```

* Start the environment
```bash
source ~/virtualenvs/acaiplus/bin/activate
```

* Install Dependencies
```bash
pip3 install -r requirements.txt
```

## Run setup

Install package locally.

```bash 
python setup.py develop
```

## Generate documentation

Install sphinx dependencies

    pip install sphinx sphinx_rtd_theme
    
Build the documentation

    cd docs && make html && cd build && python -m http.server 1234
    
View it in your browser

    http://0.0.0.0:1234/html
