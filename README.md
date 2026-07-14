# Evolutionary Model Merge

An end-to-end implementation of evolutionary model merging — automatically searching
the space of model-merge recipes (operator, mixing weights, interpolation parameters)
to combine multiple specialized neural networks into one that generalizes across
all their domains. This is the same core idea behind Sakana AI's
Evolutionary Model Merge (Jan 2024), reimplemented from scratch at small scale so it
is fully reproducible without GPU access or multi-gigabyte model downloads.

## Why this exists

Most teams that want a multi-domain model either fine-tune one giant model on
everything (expensive, and domains interfere with each other), or hand-pick merge
weights by trial and error. Evolutionary merging treats the merge recipe itself as
a genome and evolves it — no gradient descent on the merged model required, no
retraining, just search over how to combine existing weights.

## Results

| Strategy                          | Multi-domain avg accuracy |
|-----------------------------------|---------------------------|
| Single expert (math_reasoning)    | 0.2487                    |
| Naive equal-weight linear merge   | 0.4580                    |
| Single expert (japanese_nlp)      | 0.4860                    |
| Single expert (code_generation)   | 0.5000                    |
| Evolved merge                     | 0.5047                    |

The evolved merge beats naive equal-weight merging by +10.3% relative and
slightly exceeds even the strongest individual expert, without needing to know in
advance which expert was strongest. Evolution converged on a linear merge that
learned to almost entirely discount the noisiest expert (japanese_nlp, weight
about 0.01) and lean on code_generation (weight about 1.7).

## Architecture

```
src/
  models/expert_net.py      # shared architecture all experts must share to be mergeable
  utils/datasets.py         # synthetic multi-domain datasets (stand-in for real corpora)
  core/genome.py            # MergeGenome: the evolvable "recipe" representation
  core/merge_ops.py         # linear, SLERP, and TIES-style merge operators
  core/fitness.py           # decodes a genome into a real merged model and scores it
  core/evolve.py            # the (mu + lambda) evolutionary search loop
  train_experts.py          # Stage 2: trains the 3 base domain experts
  compare_baselines.py      # Stage 6: honest comparison vs naive baselines
  api/main.py                # FastAPI service wrapping the engine
tests/test_core.py           # unit tests for merge math and genome operators
```

## Running it

```bash
pip install -r requirements.txt

# Stage 1-2: train the base expert models
python src/train_experts.py

# Stage 3-5: run evolutionary search (prints generation-by-generation progress)
python src/core/evolve.py

# Stage 6: compare evolved result against naive baselines
python src/compare_baselines.py

# Run tests
python -m pytest tests/ -v

# Run the API
PYTHONPATH=src uvicorn api.main:app --reload --app-dir src
# then: curl -X POST localhost:8000/evolve -H "Content-Type: application/json" \
#         -d '{"population_size": 16, "generations": 20}'
```

## Design notes / things to discuss in an interview

- Why genomes encode recipes, not weights: keeps the search space tiny (a few
  floats) regardless of how large the underlying models are. The same trick
  makes this approach tractable on billion-parameter LLMs in the original Sakana
  AI work.
- Why TIES exists alongside linear/SLERP: naive averaging causes destructive
  interference when experts disagree on parameter sign. TIES trims low-magnitude
  weights and only merges parameters where a majority of experts agree on sign.
- Why elitism plus tournament selection: elitism guarantees fitness never
  regresses generation over generation; tournament selection avoids the premature
  convergence that pure fitness-proportionate selection can cause.
- Known limitation: small dense nets on synthetic data, run on CPU. The
  algorithm is identical to what's used on real LLMs. Swapping in real HuggingFace
  checkpoints and a real benchmark suite (MMLU, HumanEval) is the production path,
  gated by GPU and cloud budget rather than algorithm design.

## License

MIT
