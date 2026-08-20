# The base model must not be Chinese-origin, so Gemma 4 E2B is locked in

Commonwealth Bank cannot deploy Chinese-origin models. A prior prototype fine-tuned
Qwen3-4B and it scored better than anything else (dev recall 1.000 vs Gemma's 0.955),
but it is not deployable at any quality. `google/gemma-4-E2B-it` (Apache 2.0, 2.3B
effective parameters, 128K context) is the locked base model for this repo. It is
compliance-viable, already proven to train with this recipe, and smaller — so it also
improves latency and cost.

There is no model bake-off in this repo. Re-opening the model choice is out of scope.
