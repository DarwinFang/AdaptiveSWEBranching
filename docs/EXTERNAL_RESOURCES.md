# External resources on kb3

The repository references these resources in place. It does not download or copy
them.

| Resource | Frozen identity / path |
|---|---|
| SWE-smith-py snapshot | `/home/fangzhaohao/smc4swe/data/hf_snapshots/X__SWE-bench__SWE-smith-py` |
| Dataset revision | `77cab9055d42ab4a5c25c89a8f937096db13558e` |
| Dataset manifest | `/home/fangzhaohao/smc4swe/data/dataset_manifests/X.swe-smith-py.manifest.json` |
| Repository-held-out split | `/home/fangzhaohao/smc4swe/configs/splits/swesmith_py_80_10_10_seed20260722.json` |
| SWE-smith harness | `/home/fangzhaohao/smc4swe/data/evaluation_harness/SWE-smith` |
| Harness commit | `9b74ac08118a85c39c356802f7961893af73e07f` |
| Existing workspace cache | `/home/fangzhaohao/recov-runs/workspace_cache` |
| Conda environment | `/home/fangzhaohao/miniconda3/envs/openhands` |
| OpenHands SDK | `1.21.0` |
| Ollama endpoint | `http://127.0.0.1:11436` |
| Qwen model | `qwen3-coder:30b-a3b-q8_0` |
| Ollama model digest | `7b438a19895a90821e60a42ed894cd40e746082200dff6aa5a4285b529e2a4a5` |

The doctor command checks these identities before a live experiment. Container
image tags from the dataset are resolved to local Docker digests and recorded per
task/checkpoint; tags alone are never treated as immutable identities.

The shared workspace cache was created by the old recoverability project and
contains an untracked `.recov_identity.json` provenance marker. The cache is
read-only to this project. Each copied run workspace removes that marker only
after confirming Git does not track it, preventing old-project metadata and
issue text from contaminating candidate patches or agent evidence.
