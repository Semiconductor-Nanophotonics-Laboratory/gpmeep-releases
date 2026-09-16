# gpmeep v1 license, attribution, and provenance

## Project attribution

gpmeep was conceived, created, and is led by
[Dong-Joon Yi](https://iverydj.github.io/) ([GitHub](https://github.com/iverydj)).
His gpmeep-specific contribution covers the CUDA execution architecture,
Meep integration, multi-GPU strategy, numerical and performance validation,
packaging/provenance system, and release engineering. gpmeep remains a
derivative of upstream Meep: the upstream solver, APIs, copyright, author
history, and required citation are preserved and credited separately below and
in `AUTHORS` and `README.md`.

gpmeep is a derivative of Meep and is distributed under
GPL-2.0-or-later. The package installs the complete license text at
`<prefix>/share/gpmeep/LICENSE`. Release source is provided by the gpmeep Git
repository and its checksum-bound tagged source archive. Upstream Meep
copyright and citation information remains in `LICENSE`, `README.md`,
`doc/docs/License_and_Copyright.md`, and `doc/docs/Acknowledgements.md`.

The public repository contains privacy-cleaned source snapshots rather than
the private development history. Removed files are raw evidence, resume notes,
evidence anchors, and development automation, not numerical solver or package
build source. Public-facing documentation may also be updated or have local
execution paths replaced with generic examples. The original binary retains
its original source commit and SHA-256; public Git commits must not be
substituted for those identities. See `PUBLIC_RELEASE.json` and
`PUBLIC_RELEASE_POLICY.md` at the repository root.

The MPI adjoint chunk-invariance correction is a manual backport that preserves
the identity of NanoComp/meep PR 3274 commit
`897dbe06b0ed6442d34f2a940f21acbff9375523`. The gpmeep release plan records
its origin and the added independent gradient qualifications.

Every v1 package embeds `share/gpmeep/release.json`, including:

- gpmeep distribution and upstream source versions;
- the full 40-hex gpmeep source commit;
- FP32, fast-math, MPI, Scheme, and dedicated-environment policy;
- CUDA toolkit and compiled native/PTX architecture policy;
- the actually runtime-validated GPU architecture scope.

Run `gpmeep-verify-provenance` inside the installed prefix. It verifies the
Conda ownership record, rejects `pymeep` co-installation, audits every
package-owned path, type, content hash, symlink, and every stable size. Conda
prefix-rewritten text has its installed hash verified while its recorded
pre-rewrite size is disclosed separately. The verifier then checks the embedded release identity, imports
the installed `meep` module, and proves that its Python extension and mapped
`libmeep` resolve from the same prefix. `gpmeep-self-check` adds fresh CPU,
CUDA, MPI, device-identity, no-fallback, and numerical runtime evidence.
The installer also passes its independently computed package SHA-256 into the
verifier and requires it to equal the installed Conda record.

Build receipts, raw telemetry, and large validation products are not bundled
inside the binary package. Public release summaries and verifier inputs bind
the package/source identity; immutable raw evidence is retained outside the
Git repository according to the release plan.

The package builder never stages the live checkout. It creates a retained,
checksum-bound `git archive` of the exact clean source commit and builds from
an extracted copy of that archive, excluding ignored environments, caches,
and local evidence by construction.
