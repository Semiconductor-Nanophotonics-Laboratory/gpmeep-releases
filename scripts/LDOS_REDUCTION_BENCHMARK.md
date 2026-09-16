# Receipt-bound LDOS reduction/transfer benchmark

`run-ldos-reduction-benchmark.py` publishes post-build evidence for one narrow
comparison: resident CUDA LDOS reduction/transfer versus the forced legacy
full-field host LDOS fallback inside the same CUDA FDTD workload. It is not an
end-to-end FDTD, CPU-versus-GPU, general-Meep, or multi-GPU-scaling benchmark.

The output path must not exist and must be outside the authoritative source
tree. The controller verifies the build receipt and source before and after,
first re-executes the clean committed controller and every Python helper from
held file descriptors. The controller Git commit/clean state and held Python
interpreter are bound separately from the build receipt. It then holds stable
handles for the receipt-bound executable, `libmeep`, `mpiexec`, GNU `timeout`,
the ELF loader, runtime closure, and MCA configuration. The authoritative repo
must also be the controller repo; injected bootstrap metadata or a redirected
source directory cannot name some other clean Git repository.

The authoritative entrypoint is an explicitly selected system Python invoked
with `-I -S` from a sanitized environment. A direct shebang invocation is
intentionally rejected. The isolated interpreter, root-owned non-group/world-
writable system Python/runtime libraries, and the host OS are an explicit TCB;
malicious root, kernel, or host replacement is outside this scientific-
reproducibility threat model. User `PYTHONPATH`/`PYTHONHOME`, `sitecustomize`,
and loader/preload variables are not accepted or inherited by the held-FD
child.

The worker and `mpiexec` are launched through the archived ELF loader with
RPATH inhibited and a private archived-library-first search path. GNU
`timeout` is executed through its held descriptor while retaining semantic
`argv[0]=timeout`. During each lane, the controller repeatedly records actual
process command-line/mapping epochs with a bounded-gap attempt timeline.
Completion binds the exact held-FD timeout/loader/`mpiexec`/worker invocation,
the actual PRTE descendant, and exact-rank worker epochs mapping the archived
loader/worker plus strict `libmpi.so`/`libpmix.so` families. Open MPI 5 can
replace the transient launcher with PRTE in the same PID before polling; in
that normal case `mpiexec` execution is explicitly reported as inferred from
the bound invocation and downstream PRTE/workers, not as mapped-byte proof.
Worker records independently bind executable/required regular mappings from
`/proc/self/maps`; non-executable NVIDIA character devices and whitelisted
`/dev/shm`/memfd/`/dev/zero` data mappings use a separate schema. Root-owned
system code accepted through the OS TCB is post-lane hashed and retained with
its trust source. A recursive inotify sentinel and stable closure handles
reject prefix swap-and-restore.

Before and after the lanes, trusted system `nvidia-smi` records driver,
clocks, power, temperature, utilization, memory, and compute-process
contention. During every lane, the controller instead polls a held,
root-owned system `libnvidia-ml.so.1` directly on a bounded absolute-schedule
50 ms cadence.
The raw journal binds ordered device UUIDs, compute/graphics/MPS process
lists, process-utilization cursors, clocks, power, temperature, P-state,
memory, query timing, and recomputed outsider verdicts. Baseline current and
process-utilization lists must be empty; two exact worker-presence samples are
required before GO and two empty drain samples after process exit. Both
explicitly requested GPUs must also be quiescent at the `nvidia-smi` endpoint
checks.

Success uses a second, success-only completion handshake while every CUDA
context is still resident. After all five fixed attempts and result evidence
are complete, each rank atomically publishes an exact `RESULTS_READY` file
bound to its original PID/start-time epoch, device UUID, nonce, and worker
monotonic timestamp. The controller records the exact named file identity,
then requires a subsequent current NVML/affinity sample with all ranks still
present before atomically publishing `RELEASE`. GO and RELEASE use complete
same-directory temporary inodes plus no-replace atomic publication, so a
reader never observes partial gate bytes. The sealed order is exactly
`ready -> GO -> RESULTS_READY(all ranks) -> RELEASE -> worker-exit(all ranks)
-> process-exit`. Missing or ambiguous steps cannot produce COMPLETE.
The controller is the sole journal writer, so every timestamped lifecycle
event and every sample interval must be globally non-overlapping in journal
order; exact endpoint equality is allowed. All nanosecond fields are plain
uint64 values. A sample begins only after same-poll worker-exit and
RESULTS_READY observations have been recorded.
Before GO, all rank READY files are loaded and journaled before the telemetry
origin or cadence index is initialized. A missing or partial READY matrix
returns without sampling host/NVML state, so every sealed sample has the exact
full-rank worker-state schema and every READY event precedes the first sample.

After RELEASE, if a completed NVML sample first observes an assigned compute
row absent while its pidfd is not yet ready, the controller must observe
readiness of that exact descriptor-bound PID epoch within 100 ms of the NVML
sample-finished observation. This is an observation-to-observation bound; it
does not claim the physical context-disappearance time or physical process-exit
time. If pidfd readiness is observed first, the first subsequent empty NVML
sample is still latched, but the reverse delta is a sampling-delay diagnostic,
not a physical linger bound. The report records both observation endpoints,
an explicit `absence-observed-first` or `pidfd-observed-first` order, and only
the delta matching that order (the opposite delta is null). Actual departed
context acceptance remains controlled by exact sampled PID/epoch rows and the
separate 100 ms departed-context sample gate. That gate is computed from the
fresh per-rank timestamp taken immediately after a positive pidfd poll to the
completed NVML sample's `finished_monotonic_ns`; neither callback entry nor
sample start is an endpoint. Each ready pidfd endpoint is journaled before its
deadline verdict, and still-unready ranks are rechecked against a fresh clock
after the rank loop. Any changed binding or context
reappearance after the first observed absence fails the lane. Rank-1 and
rank-2 lanes derive these rules independently for every participating rank.
The held NVML library is copied once to the fixed archive name
`archive/system/lib/libnvidia-ml.so.1`; every lane cross-binds the original
held file to that archive by size and SHA-256. Live lane completion re-verifies
both open handles. Resolution uses a platform- and WSL-aware exact bounded
allowlist. For each native platform the deterministic non-WSL order is its
`/usr/lib/<triplet>/libnvidia-ml.so.1`, then
`/lib/<triplet>/libnvidia-ml.so.1`, followed by
`/usr/lib64/libnvidia-ml.so.1`, `/lib64/libnvidia-ml.so.1`,
`/usr/lib/libnvidia-ml.so.1`, and `/lib/libnvidia-ml.so.1`; triplets are
`x86_64-linux-gnu` and `aarch64-linux-gnu`. Standard WSL candidates are appended off WSL and
preferred on WSL. Candidates are examined in deterministic order and skipped
unless that candidate's still-open descriptor itself passes the immutable
root-owned system-runtime TCB plus bounded native machine, pointer-width ELF
class, byte-order encoding, exact program-header table, loadable image,
`PT_DYNAMIC` containment, allocated-section mapping/permissions, SONAME,
build-ID, and required-export validation. Static acceptance is followed by a
five-second single-threaded child-process probe that loads the exact held FD
and resolves every required NVML symbol. A signal, timeout, load error, or
missing symbol closes that candidate and advances to the next allowlisted
path. The parent opens a pidfd immediately after `fork`; a one-byte pipe marks
successful child `setsid()` before group termination is permitted. Timeout and
exception cleanup first signals the exact unreaped child, then its ready
process group. Exact-child termination and `WNOHANG` reap are retried only to
the fixed probe deadline; cleanup never enters an unbounded blocking
`waitpid`. A Python `waitpid` interruption is cross-checked through an
independent libc nonblocking wait, every remaining interruption returns to the
monotonic deadline, and a child that cannot be proven reaped fails the
candidate.
The controller saves and temporarily enables Linux child-subreaper state before
forking. On success and failure alike it repeatedly discovers every new
adopted descendant relative to the empty pre-probe baseline, binds each epoch
with a pidfd, sends `SIGKILL`, and reaps to three consecutive empty rounds.
If primary task-children enumeration raises, containment independently scans
`/proc/*/stat` and derives the descendant graph from PPid records; an
observation failure can never count as an empty round. A tracked pidfd remains
owned until close succeeds or `EBADF` proves it closed. If Python `os.close`
fails, an independent libc close is attempted and its result is verified
before ownership is released; the same rule covers the exact-child and pipe
descriptors. A successful Linux close return is authoritative and is never
followed by a reuse-prone probe. Before the primary close, cleanup atomically
duplicates the descriptor with `F_DUPFD_CLOEXEC` as a non-reusable
open-file-description generation guard and immediately records ownership. If
the primary close raises, Linux `kcmp(KCMP_FILE)` compares the current number
with that held guard before any raw fallback: a different or closed generation
is never closed. Unresolved candidate/guard pairs remain in a process-scope
ownership registry whose guard FD has never been submitted to close. The
registry is drained again before subreaper restoration and before another NVML
candidate can be probed, and prevents resolver continuation while unresolved.
If closing the sole guard itself has an ambiguous Python exception, its bare
number is poisoned and is never retried while that descriptor generation
remains unresolved; the resolver is barred from continuing, so reuse cannot
cause an unrelated FD to be closed. A later successful, explicitly tracked
kernel allocation (`pidfd_open` or atomic `F_DUPFD_CLOEXEC`) at that same
number proves that the ambiguous older generation is gone. The new owned
generation then rebinds the number and clears the stale poison before it is
used or closed. Poison is enforced at every close-helper and raw-close entry
for all other paths. It also
invalidates the corresponding local pidfd record immediately, forcing
PID/start-epoch kill/reap fallback and preventing later signal, poll, or close
of the unsafe number. Raw close setup, call, return, errno retrieval, and error
decoding form one total exception boundary, so post-call failures always reach
the poison transition.
Every descriptor-allocating operation in this probe path (generation-guard
duplication, pidfd acquisition, and readiness-pipe creation) is routed through
a tiny receipt-bound native allocation shim. The exact-lock C compiler builds
the shared object during the authoritative build; the qualification sentinel
monitors it and the build receipt binds it. The controller first validates and
holds that exact receipt artifact, then loads it through its `/proc/self/fd`
path before any nvidia-smi or NVML candidate probe can allocate a descriptor.
After those resolvers identify the remaining archive inputs, the controller
archives the held shim and immediately proves that the copy has the loaded
artifact's exact size and digest. Each ABI
entry point receives caller-owned result storage, initializes all ownership
fields before the syscall, records the returned descriptor(s) and errno, and
publishes `completed` last with release ordering. Consequently, even a Python
`BaseException` dispatched immediately after the native call returns leaves
the allocated generation visible to the caller. A successful result is wrapped
in a temporary native-owned integer whose finalizer retains ownership across
every helper return/assignment boundary; an explicit verified/raw close
disarms that exact owner. Exceptions raised inside an allocation frame close
only that frame's recorded result. They are not inferred from `sys.exc_info()`,
so an unrelated exception currently handled by an outer frame cannot make a
successful nested allocation close itself. Real `sys.settrace` regressions
inject at the first post-call line and at the guard, pidfd, and pipe helper
return events, then require exact descriptor-baseline restoration. Every
recorded fresh descriptor is raw-closed once, ambiguous cleanup is poisoned,
and the original allocation exception remains primary. This design imposes no one-native-task
assumption and therefore remains valid after NVML or the GPU driver creates a
persistent controller task. A built-in launch-time self-test exercises all
three entry points and proves that their descriptor set returns to its exact
baseline.

The temporary owner's finalizer never retries a bare descriptor number after
its close helper raises. The helper may already have returned an authoritative
kernel close and disarmed the owner before an asynchronous Python exception;
the numeric slot can then refer to a different generation. A still-armed owner
is instead invalidated and the number poisoned, while an already-disarmed owner
requires no further action. A real trace regression closes the original,
reallocates the same numeric slot for `/dev/zero` at the helper-return boundary,
raises, and proves the replacement remains open and the controller state clean.
Every raw-close entry also treats a disarmed `NativeOwnedFd` as already
resolved, so a caller retry after the same helper-return exception cannot close
the reused numeric slot. A second real trace regression exercises that retry.

The C build statically asserts every result-structure size and field offset.
The loaded library also exports packed fingerprints computed from its actual
`sizeof`/`offsetof` values; the controller compares both fingerprints with its
ctypes layouts before the first allocation. Thus an ABI/layout mismatch fails
at load time rather than being inferred from matching controller constants.

The controller then validates the semantic identity of every successful
allocation before accepting ownership. A generation guard must be CLOEXEC and
`kcmp(KCMP_FILE)`-equal to its source open-file description. A pidfd must report
the requested `Pid`/`NSpid` identity and the target's `/proc` start-time epoch
must equal the expected epoch both before and after allocation. Readiness-pipe
endpoints must be distinct CLOEXEC FIFO descriptors for the same pipe with the
correct read/write access modes. Any unavailable, inconsistent, or mismatched
identity fails closed and closes the newly recorded descriptor generation.
If descendant pidfd acquisition or signaling fails, the adopted unreaped PID
is retained with its `/proc` start-time epoch, directly killed, and still
passed through interruption-safe `waitpid` until it is reaped. Repeated
diagnostics are de-duplicated.
Any observed descendant rejects the candidate even after successful cleanup;
survivors, cleanup errors, or subreaper-restore errors also fail closed while
preserving the primary error. Readiness, signal, poll, wait, close,
containment, and restore stages each collect `BaseException` independently so
no optional cleanup failure can skip the later mandatory stages.
Containment also owns an unconditional fixed-count emergency finalizer which
continues PID discovery, kill, nonblocking reap, and verified descriptor drain
without relying on the clock or sleep path that may have failed.
The outer NVML probe uses the same rule: cleanup-deadline clock failure selects
a fixed-count exact-child reap path, and descriptor close, descendant
containment, retained-capability drain, and subreaper restoration are isolated
stages which cannot skip their successors.
Child-subreaper restoration retries both the normal setter and an independent
raw `prctl`, then queries the kernel for the exact prior state; unresolved
state prevents a later candidate from running.
The fixed archived NVML copy repeats the same probe before sampling.
The physical-evidence controller has an explicit 64-bit LP64 userspace floor
and fails before probing candidate paths on any other ABI. It does not claim
or silently remap a 32-bit compatibility userspace.
Both supported LP64 machines use the same bounded WSL candidate paths; native
ELF validation on the held descriptor rejects a candidate for the wrong
architecture.
No selected path is reopened. Distinct `/usr/lib64` and `/lib64` layouts are
explicit TCB roots. `nvidia-smi` likewise uses only the
fixed `/usr/bin`, `/usr/local/bin`, and WSL paths; WSL is detected from the
bounded kernel-release file and its tool path is preferred, while every tool
must pass immutable-file and native machine/class/encoding validation on the
same descriptor used for archiving and execution. The selector reads the
complete native ELF header, requires `ET_EXEC` or `ET_DYN`, validates a
bounded exact program-header table, and requires a file-bounded executable
`PT_LOAD` containing the nonzero entry point before a candidate can shadow a
later path. Every auxiliary `PT_DYNAMIC` or `PT_INTERP` must map exactly into
a permission-compatible `PT_LOAD`; `PT_INTERP`, when present, is unique,
bounded, NUL-terminated, and an absolute normalized ASCII path. That exact
interpreter is independently opened, required to pass the native immutable
system-tool TCB contract, and held with the executable. It must itself be a
native `ET_DYN` image with exactly one dynamic segment, at least one load
segment, and no nested `PT_INTERP`, rather than merely any exit-zero utility.
Candidate selection also requires a bounded successful `--help` startup
through the held interpreter whose C-locale output contains the NVIDIA System
Management Interface heading, the `nvidia-smi [OPTION` usage grammar, and the
`--help` option, with empty stderr. Telemetry invokes the archived executable
through a duplicate of that same held interpreter, and repeats the startup
probe before its first query. The final report binds the source and duplicate
interpreter size/SHA-256/fingerprint records, native ELF identity, and immutable
system-tool TCB result. The raw offline
header contains only the archive-relative content identity: size, SHA-256,
ELF class/encoding/machine, entry point, program/load counts, DT_SONAME, GNU
build ID, and the required defined NVML exports. Standalone validation parses
those bounded ELF structures without loading the library. Every nonempty
allocated section must have an exact address/file mapping and compatible
permissions in a `PT_LOAD`, while `SHT_DYNAMIC` must also be contained by the
sole `PT_DYNAMIC`. Dynamic semantics
stop at the first `DT_NULL`, with exactly one preceding `DT_SONAME`; required
exports must use a supported defined section index, and the GNU build-ID note
must use canonical `namesz=4`/`GNU\0` encoding. Explicit file-size, section-
count, aggregate parse-work, string-scan, and metadata-size budgets plus
non-overlap of distinct relevant file-backed sections prevent compact hostile
ELFs from amplifying parser work. This offline check does not claim to
re-prove the original host path or inode metadata. Full original StableFile
path/fingerprint provenance remains in the live outer report and archive
manifest, both bound by the final artifact manifest/terminal. Before a FAILED
terminal can bind partial archive entries, regular files and the no-follow
directory tree are fsynced leaf-first.
The two-rank MPI app contexts select logical CUDA ordinals 0 and 1 from the
UUID-ordered `CUDA_VISIBLE_DEVICES` namespace, and each worker must report the
matching distinct physical CUDA UUID.

The controller and every MPI rank use distinct complete physical cores. The
controller pins all of its native TIDs with a bounded fixed-point protocol,
passes each rank an exact Linux CPU list, verifies every controller/PRTE/worker
TID at each scheduled poll, and restores both original and newly arrived TIDs
before publication. Each C++ update records monotonic start/stop/elapsed time,
CPU before/after, affinity, context-switch deltas, and page-fault deltas. The
parser recomputes segment aggregates and binds pair seconds to the maximum
per-rank sum of those raw intervals; CPU migration, involuntary switches,
major faults, aggregate mismatches, and excessive external CPU/PSI contention
are fail-closed.

The prior `cadacc0e` receipt predates the raw-record/runtime-attestation
contract and is intentionally not accepted for final publication. After the
updated C++ worker is merged and a new authoritative receipt is created, use:

```sh
DEVICE_UUIDS=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
DEVICE_UUIDS="${DEVICE_UUIDS},GPU-11111111-2222-3333-4444-555555555555"
REPO=/absolute/path/to/gpmeep
NEW_RECEIPT_ID=replace_with_new_64_lowercase_hex_receipt_id
CONTROLLER_PYTHON=/usr/bin/python3
env -i HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin \
"${CONTROLLER_PYTHON}" -I -S "${REPO}/scripts/run-ldos-reduction-benchmark.py" \
  --authoritative-repo "${REPO}" \
  --build-receipt "${REPO}/build/meep-cuda-mpi-python-fp32/build-provenance.json" \
  --expected-receipt-id "${NEW_RECEIPT_ID}" \
  --mpiexec "${REPO}/.envs/meep-gpu-cuda-mpi/bin/mpiexec" \
  --device-uuids "${DEVICE_UUIDS}" \
  --output /path/outside/source/m19-ldos-reduction-transfer
```

`--device-uuids` is mandatory so a clean environment never overrides a
scheduler allocation by silently choosing system-global device indices. A
successful directory contains raw stdout/stderr and combined logs, exact
command/environment attempts, an append-only canonical JSONL telemetry
journal (including each completed poll and exact ready/GO/worker-exit/root-exit
ordering plus the `RESULTS_READY`/`RELEASE` bindings), parsed lane JSON,
archived inputs, the mutation-sentinel record,
`summary.json`, `report.json`, `report.md`,
`artifacts.sha256.json`, `COMPLETE`, and finally `TERMINAL`.
`TERMINAL` is the sole final authority and binds either `COMPLETE` or `FAILED`.
A missing `TERMINAL` means the publication was interrupted.
After all live handles and observers close, success publication reopens the
fixed NVML archive through its current path and cross-checks its size, SHA-256,
ELF identity, report/archive-manifest provenance, and every sealed raw-journal
header. It then reopens the finalized artifact manifest and all bound current
files immediately before `COMPLETE` and again before and after `TERMINAL`.
A path swap at any of those publication boundaries invalidates success and is
recovered as an authoritative `FAILED` terminal while `RUNNING.json` is still
present or is safely recreated. Success deliberately retains the exact current
`RUNNING.json` through the durable `TERMINAL` write and all post-terminal
manifest/archive/report/journal/authority checks. Only then is `RUNNING.json`
unlinked and its directory fsynced, followed by a final current-path authority
check. Thus a crash before the terminal remains visibly RUNNING, while a crash
during marker retirement already has a durable, verified terminal authority.
An exception during a scheduled NVML/host query fails the whole lane and is
published through the journal's error terminal; it is not represented as a
successful sample or a synthetic "failed-query sample." Because observations
are discrete 50 ms polls, same-UID interference that starts and finishes
entirely between two ticks is outside this sampling/trust boundary. Endpoint
checks, empty baselines, pre-GO presence checks, post-exit drains, and every
completed in-lane poll narrow that boundary but do not eliminate it.

Each lane makes exactly five alternating AB/BA attempts; there is no in-run
retry, replacement, or favorable-sample selection. Every accepted attempt
contains pair records with raw
host/resident seconds, independently recomputable speedup, nine complex LDOS
samples, FNV-1a digests, and maximum absolute/relative error. Every rank emits
exact LDOS CPU/CUDA/kernel/result-transfer counters and D2H bytes for every
pair. The controller recomputes the pair values, errors, digests, minimum, and
median. For every repetition, every rank's first segment must stop before its
second segment starts, and the collective maximum first-segment stop across
ranks must not exceed the collective minimum second-segment start. The
collective end of repetition `i` must not exceed the collective beginning of
repetition `i+1`; rank/pair/runtime/summary/PASS records also follow the exact
rank-ordered physical stream topology. Every rank's first update is not
earlier than the controller's pre-publication GO lower bound, and its final
update is not later than that same rank's `RESULTS_READY` timestamp or process completion. Every pair
must exceed `1.20x`. This remains a narrow LDOS
reduction/transfer claim, not an end-to-end speed claim.
All five host sample digests and all five resident sample digests must also be
bitwise identical across repetitions; tolerance-only within-pair agreement is
not enough for the deterministic label.

If any attempted rank observes a major fault or involuntary context switch,
all ranks first emit a rank-ordered `gpmeep-ldos-rejected-v1` diagnostic for
that exact attempt and the fixed run exits nonzero. The rejected matrix binds
all 16+16 raw update intervals per rank, aggregate scheduling/fault counters,
D2H and LDOS dispatch/kernel/result counters, global reduction seconds, and
physics samples/digests/errors. Minor faults remain recorded diagnostics and
are not rejection predicates. Before rejected repetition `N`, stdout must
contain exactly the accepted `0..N-1` rank matrices in rank order followed by
their AB/BA pair rows. Those prior records pass the same complete canonical
schema, type, counter, transfer, interval, aggregate-second, sample/digest,
error, device, affinity, per-rank order, and collective-order validator as a
successful lane. Because the worker applies the `1.20x` performance gate only
after all five attempts complete, that final five-sample acceptance gate is
not imposed on a structurally valid prior prefix of a scheduling-rejected run;
the whole run remains failed. The rejected attempt obeys the same per-rank and
collective AB/BA timestamp order, begins after the preceding accepted prefix,
and is cross-bound between controller GO and process completion before it can
be persisted as scientific failure evidence. Missing controller endpoints
keep the worker failure primary but suppress unbound benchmark evidence.
Rank/pair/rejected structured evidence is stdout-only;
the focused PASS, benchmark, runtime mapping, standalone or tagged `COMPLETE`
authority token, and unknown
`gpmeep-*` markers are forbidden anywhere in either stream, while unrelated
non-gpmeep diagnostics and PASS lines remain diagnostic-only. A rejected
record is incompatible with exit 0 and successful performance statistics. A separate physical
rerun, if desired, must use a separate output directory.
Both the pre-parser worker-primary classifier and the final success parser use
the same failure-word scan over unstructured diagnostic lines. Canonical
structured JSON records are instead validated field by field, so an
ordinary string value such as `/scratch/failed-runs/...` cannot create a false
failure solely because of a path component.

Process containment is Linux-only and fail-closed: a child subreaper plus
pidfds tracks and reaps process trees, including `setsid()` descendants. GNU
timeout exit 124/137, Python-side timeout, surviving descendants, observation
gaps, or incomplete PRTE/PMIx proof all fail publication. All publication
directories are fsynced before the terminal authority marker. Failure
classification preserves a worker's nonzero status or explicit `FAIL` as the
primary error before best-effort pidfd/NVML teardown diagnostics; cleanup
cannot replace the scientific failure reason, even if the teardown diagnostic
routine itself raises; rejected evidence is still persisted first. If a
callback/observer failure
itself terminates a still-live process, that observation failure remains the
primary. An unexpected process-control exception which terminates a live child
is likewise primary for a bare induced nonzero exit. A fully validated rejected
matrix or explicit scientific FAIL emitted before that exception remains an
independent worker primary; so does a worker that had already exited. A nominal success still must
complete and independently validate the full handshake and teardown.
Failure publication never follows or recursively deletes hostile entries: conflicting
terminal/authority paths are atomically renamed to recoverable `superseded-*`
names, partial regular files are hashed through `O_NOFOLLOW` descriptors, and
`RUNNING.json` is retired only after the FAILED terminal and authority have
been durably reverified through single-link, no-follow current-path handles.
If either name is swapped at that boundary, the suspect entry is quarantined,
the outside target is untouched, and a valid RUNNING marker is retained so
failure publication can be retried.

Run the GPU-free parser, bootstrap-spoof, mutation/swap-restore,
process-containment, output-claim, archive, and synthetic controller tests
with:

```sh
python3 -m unittest scripts.tests.test_ldos_reduction_benchmark -v
```
