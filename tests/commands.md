## CLI Commands

This doc provides a detailed list of commands available in the repo as well as how to custom them.

### Pipeline Commands

| Command | Basic | Override Config Example | Custom Config |
|---|---|---|---|
| **auto_agent** | `python -m Fuser.auto_agent problem=/path/to/problem.py` | `python -m Fuser.auto_agent problem=/path/to/problem.py ka.model=gpt-5 router.model=gpt-5 routing.allow_fallback=false` | `python -m Fuser.auto_agent --config-name custom_auto_agent problem=/path/to/problem.py` |
| **pipeline** | `python -m Fuser.pipeline problem=/path/to/problem.py` | `python -m Fuser.pipeline problem=/path/to/problem.py extractor.model=gpt-5 dispatcher.model=o4-mini composer.model=o4-mini` | `python -m Fuser.pipeline --config-name custom_pipeline problem=/path/to/problem.py` |
| **e2e_test** | `python e2e_test.py` | `python e2e_test.py num_workers=4 max_rounds=10 model_name=gpt-5 high_reasoning_effort=true` | `python e2e_test.py --config-name custom_e2e_test` |


#### Component Commands
| Command | Basic | Override Config Example | Custom Config |
|---|---|---|---|
| **cli** (orchestrator) | `python -m Fuser.cli problem=/path/to/problem.py` | `python -m Fuser.cli problem=/path/to/problem.py model=gpt-5 workers=4 max_iters=10 stream=winner` | `python -m Fuser.cli --config-name custom_fuser problem=/path/to/problem.py` |
| **subgraph_extractor** | `python -m Fuser.subgraph_extractor problem=/path/to/problem.py` | `python -m Fuser.subgraph_extractor problem=/path/to/problem.py model=gpt-5 workers=4 max_iters=5` | `python -m Fuser.subgraph_extractor --config-name custom_subgraph_extractor problem=/path/to/problem.py` |
| **dispatch_kernel_agent** | `python -m Fuser.dispatch_kernel_agent subgraphs=/path/to/subgraphs.json` | `python -m Fuser.dispatch_kernel_agent subgraphs=/path/to/subgraphs.json agent_model=gpt-5 out_dir=./kernels_out jobs=2` | `python -m Fuser.dispatch_kernel_agent --config-name custom_dispatch subgraphs=/path/to/subgraphs.json` |
| **compose_end_to_end** | `python -m Fuser.compose_end_to_end problem=/path/to/problem.py subgraphs=/path/to/subgraphs.json kernels_summary=/path/to/summary.json` | `python -m Fuser.compose_end_to_end problem=/path/to/problem.py subgraphs=/path/to/subgraphs.json kernels_summary=/path/to/summary.json model=gpt-5 verify=true` | `python -m Fuser.compose_end_to_end --config-name custom_compose problem=/path/to/problem.py subgraphs=/path/to/subgraphs.json kernels_summary=/path/to/summary.json` |


### UI Commands

| UI | Entry Point (Default) | Entry Point (Custom Port) | Script (Default) | Script (Custom Port) |
|---|---|---|---|---|
| **kernel-agent** | `kernel-agent` | `kernel-agent port=8086` | `python scripts/kernel-agent.py` | `python scripts/kernel-agent.py port=8086` |
| **triton-ui** | `triton-ui` | `triton-ui port=8085` | `python scripts/triton_ui.py` | `python scripts/triton_ui.py port=8085` |
| **fuser-ui** | `fuser-ui` | `fuser-ui port=8084` | `python scripts/fuser_ui.py` | `python scripts/fuser_ui.py port=8084` |
