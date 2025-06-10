# verl documentations

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d _build/html/
```
Launch your browser and navigate to http://localhost:8000 to view the documentation.
