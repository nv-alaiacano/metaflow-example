# Running the demos

This should run the demos for you. First we need to set up the requirements:

```
pip install -r requirements.txt
```

Then we can run them with `python <file> run`.

```
python examples/feast/01-generate-synthetic-data.py run
```

Metaflow will validate that the DAGs are valid before executing.
