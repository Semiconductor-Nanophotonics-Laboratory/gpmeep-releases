# Exact development environments

The `*-linux-64.lock` files are Micromamba explicit specifications generated
from the corresponding environment YAML and a validated local environment.
Every package URL includes its SHA-256 digest.

`scripts/create-env.sh` uses these files on Linux x86_64 and verifies that an
existing prefix contains exactly the locked package set. The YAML files remain
the human-maintained statement of dependency intent and the fallback for
platforms without a committed exact lock.

Lock updates are intentional maintenance operations: resolve a fresh
environment from the YAML, run the CPU and CUDA validation suites, and only
then replace the explicit lock with the output of:

```sh
micromamba list --prefix <validated-prefix> --explicit --sha256
```
