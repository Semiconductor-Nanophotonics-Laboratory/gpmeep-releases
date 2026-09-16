from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import re
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "scripts/user-workloads"
sys.path.insert(0, str(WORKLOADS))


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load("gpmeep_v103_common", WORKLOADS / "common.py")
sys.modules["common"] = COMMON
MATRIX = load(
    "gpmeep_v103_matrix",
    WORKLOADS / "run_installed_aunp_performance_matrix.py",
)
VERIFY = load(
    "gpmeep_v103_verify",
    ROOT / "scripts/verify-gpmeep-package.py",
)
SELF_CHECK = load(
    "gpmeep_v103_self_check",
    ROOT / "scripts/gpmeep-self-check.py",
)

PERFORMANCE_DOC = "doc/docs/GPMEEP_PERFORMANCE.md"
README_DOC = "README.md"
HISTORICAL_RELEASE_NOTES_DOC = "doc/docs/GPMEEP_V1_0_1_RELEASE_NOTES.md"
CURRENT_RELEASE_NOTES_DOC = "doc/docs/GPMEEP_V1_0_3_RELEASE_NOTES.md"
CROSSOVER_DOC = "doc/docs/GPMEEP_GPU_CROSSOVER.md"
PACKAGED_DOCS = (
    "GPMEEP_INSTALLATION.md",
    "GPMEEP_FEATURE_MATRIX.md",
    "GPMEEP_PERFORMANCE.md",
    "GPMEEP_GPU_CROSSOVER.md",
    "GPMEEP_V1_0_1_RELEASE_NOTES.md",
    "GPMEEP_V1_0_2_RELEASE_NOTES.md",
    "GPMEEP_V1_0_3_RELEASE_NOTES.md",
    "GPMEEP_LICENSE_AND_PROVENANCE.md",
)
ACTIVE_IDENTITY_MARKERS = {
    "packaging/conda/meta.yaml": (
        '{% set distribution_version = "1.0.3" %}',
        "{% set build_number = 0 %}",
    ),
    "packaging/conda/release.json.in": (
        '"distribution_version": "1.0.3"',
    ),
    "scripts/build-conda-package.sh": ('DISTRIBUTION_VERSION="1.0.3"',),
    "scripts/gpmeep_release_manifest.py": ('DISTRIBUTION_VERSION = "1.0.3"',),
    "scripts/verify-gpmeep-package.py": (
        "CONDA_BUILD_NUMBER = 0",
        'CONDA_BUILD_STRING = "cuda124_mpi_openmpi_py311_0"',
    ),
    "scripts/user-workloads/common.py": (
        'GPMEEP_DISTRIBUTION_VERSION = _DISTRIBUTION_VERSION',
    ),
    "README.md": ("gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",),
    "doc/docs/GPMEEP_INSTALLATION.md": (
        "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",
    ),
    "doc/docs/GPMEEP_V1_0_3_RELEASE_NOTES.md": (
        "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",
    ),
}
FORBIDDEN_ACTIVE_IDENTITIES = (
    "gpmeep-1.0.1-cuda124_mpi_openmpi_py311_1",
    "gpmeep-1.0.2-cuda124_mpi_openmpi_py311_1",
    "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_1",
    "cuda124_mpi_openmpi_py311_1",
    "CONDA_BUILD_NUMBER = 1",
    "{% set build_number = 1 %}",
)
README_ARCHITECTURE_HEADING = "How GPU acceleration works"
README_ARCHITECTURE_SUBSECTIONS = (
    "Backend selection is owner-wide and state-atomic",
    "Persistent field residency and coherence",
    "CUDA FDTD pipeline and supported physics",
    "DFT monitors and scientific reductions",
    "MPI and multi-GPU execution",
    "FP32, reductions, and numerical accuracy",
    "Adjoint, CW, and host-side work",
    "Implementation source map",
)
README_ARCHITECTURE_REQUIRED_PROSE = {
    None: (
        "The acceleration layer sits below the existing `fields::step()` pipeline",
        "persistent mirror cache per fields_chunk",
        "selective host publication only when it is required",
    ),
    "Backend selection is owner-wide and state-atomic": (
        "CPU is the default",
        "entirely CPU or entirely CUDA for that fields owner",
        "the reference floor is 524,288 local cells",
        "1,024 bytes per local cell plus 64 MiB",
        "designed to reject the bounded launch-dominated cases used to calibrate it",
        "it is not a guarantee that the selected backend is faster",
    ),
    "Persistent field residency and coherence": (
        "Meep's host arrays remain the API-visible storage identity",
        "separate owner-level registries",
        "ends with `finish(false)`",
        "A selective point read can copy one requested scalar",
        "Generic host operations or callbacks that directly read or mutate host field arrays",
        "every dirty D2H copy is first placed in staging",
    ),
    "CUDA FDTD pipeline and supported physics": (
        "B curl, B sources, and B boundaries",
        "D curl, D sources, and D boundaries",
        "Cylindrical coordinates and BFAST each have supported CUDA paths",
        "automatic multi-monitor DFT batching is fail-closed in v1.0.2",
        "An arbitrary custom susceptibility subclass is not implicitly CUDA-capable",
    ),
    "DFT monitors and scientific reductions": (
        "For scalar/spectral reductions such as flux, energy, force, and LDOS",
        "Requested-array materialization and checkpoint/output instead use bounded staging",
        "HDF5 file I/O is host work",
        "MPB samples eigenmode profiles on the host",
        "applies to the validated FDTD time step, not to every output API",
    ),
    "MPI and multi-GPU execution": (
        "Absent an explicit `MEEP_GPU_DEVICE` or `mp.gpu.select_device(...)` selection",
        "stable physical/MIG identities are then validated collectively",
        "one Python rank does not transparently divide one domain over every GPU",
        "Direct device-buffer `MPI_Isend`/`MPI_Irecv` is used only when every rank",
    ),
    "FP32, reductions, and numerical accuracy": (
        "CPU and CUDA lanes with Meep single precision",
        "CUDA fast math is disabled",
        "based on scientific tolerances, not bitwise identity",
    ),
    "Adjoint, CW, and host-side work": (
        "The complete inverse-design program is not GPU-resident",
        "Python orchestration, JAX/Autograd/NLopt",
        "continuous-wave solver has a qualified resident FP32 CUDA BiCGSTAB-L path",
        "Small, short, monitor-heavy, host-heavy, or excessively decomposed jobs can remain slower than CPU",
    ),
    "Implementation source map": (
        "[`python/meep.i`](python/meep.i)",
        "[`src/gpu_backend.cpp`](src/gpu_backend.cpp)",
        "[`src/step.cpp`](src/step.cpp)",
        "[`src/step_db.cpp`](src/step_db.cpp)",
        "[`src/dft.cpp`](src/dft.cpp)",
        "[`src/mympi.cpp`](src/mympi.cpp)",
        "[`cuda/src/runtime.cu`](cuda/src/runtime.cu)",
    ),
}
README_ARCHITECTURE_FORBIDDEN_OVERCLAIMS = (
    "Every workload is faster on CUDA.",
    "Every output API executes on CUDA.",
    "Every output API executes on the GPU and CUDA is always faster.",
    "One Python rank transparently uses all visible GPUs.",
    "The CPU lane is FP64 and the CUDA lane is FP32.",
    "The complete inverse-design program runs on the GPU.",
    "The complete inverse-design program is GPU-resident.",
    "The complete inverse-design program is fully GPU resident.",
    "All Meep features run on CUDA.",
    "All sm60+ GPUs are runtime-qualified.",
)
README_ARCHITECTURE_FORBIDDEN_PATTERNS = (
    r"\bevery (?:workload|simulation|job)\b[^.]{0,100}\b(?:faster|speedup)\b",
    r"\bevery output api\b[^.]{0,100}\b(?:gpu|cuda)\b",
    r"\b(?:cuda|gpu)\b[^.]{0,100}\balways faster\b",
    r"\bone python rank\b[^.]{0,100}\btransparently\b[^.]{0,100}\b(?:all|multiple)\b[^.]{0,100}\bgpu",
    r"\bcpu lane\b[^.]{0,60}\bfp64\b[^.]{0,100}\bcuda lane\b[^.]{0,60}\bfp32\b",
    r"\bcomplete inverse-design program\b[^.]{0,60}(?:\bis\s+(?!not\b)(?:(?:entirely|fully|wholly)\s+)?(?:gpu|cuda)(?:[- ]resident)?\b|\b(?:runs|executes)\s+(?:(?:entirely|fully|wholly)\s+)?on\s+(?:the\s+)?(?:gpu|cuda)\b)",
    r"\ball meep features\b[^.]{0,100}\b(?:gpu|cuda)\b",
    r"\ball sm60\+ gpu[^.]{0,100}\bruntime-qualified\b",
)
README_ARCHITECTURE_MINIMUM_WORDS = 1000
DOC_CONTRACTS = {
    PERFORMANCE_DOC: (
        "| CPU8, one physical core per MPI rank | 163.823542 s | 178.197563 s | 1.0000x | 1.0000x |",
        "| GPU1 | 16.680281 s | 41.041740 s | 9.8214x | 4.3419x |",
        "| GPU2 | 10.951151 s | 30.025477 s | 14.9595x | 5.9349x |",
        "GPU2/GPU1 FDTD scaling is 1.5232x and end-to-end scaling is 1.3669x.",
        "The authoritative terminal is source commit\n"
        "`adc67c3466774bbdb1875d996e4e0e50aeb7a845`.",
        "Its `report.json`, `COMPLETE`,\n"
        "and final adversarial-audit SHA-256 values are respectively\n"
        "`50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc`,\n"
        "`7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f`,\n"
        "and `4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639`.",
    ),
    HISTORICAL_RELEASE_NOTES_DOC: (
        "| CPU8 physical ranks | 163.823542 s | 178.197563 s | 1.0000x | 1.0000x |",
        "| GPU1 | 16.680281 s | 41.041740 s | 9.8214x | 4.3419x |",
        "| GPU2 | 10.951151 s | 30.025477 s | 14.9595x | 5.9349x |",
        "GPU1-to-GPU2 scaling is 1.5232x for FDTD and 1.3669x end to end.",
        "- source commit: `adc67c3466774bbdb1875d996e4e0e50aeb7a845`",
        "- report SHA-256:\n"
        "  `50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc`",
        "- COMPLETE SHA-256:\n"
        "  `7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f`",
        "- final adversarial-audit SHA-256:\n"
        "  `4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639`",
    ),
    CURRENT_RELEASE_NOTES_DOC: (
        "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",
        "v1.0.3 is a field-feedback compatibility and packaging patch.",
        "It does not\nchange the accepted CUDA FDTD numerical kernels or their FP32 policy.",
        "explicitly targets the conda-forge glibc 2.17 C sysroot",
        "Every gpmeep-owned ELF is audited during the build",
        "same-version artifact under `broken/` is rejected",
        "Explicit CUDA architecture lists such as `86-real;86-virtual` are supported",
        "must include at least one native `-real`\n  target",
        "The requested policy must exactly match the\n  recorded native and PTX inventories.",
        "These measurements are production observations, not\nportable v1.0.3 performance promises.",
        "v1.0.3 includes no speculative kernel\nor MPI transport rewrite",
        "clean Ubuntu 22.04/glibc 2.35 installation and import",
        "native CUDA-code inventory",
    ),
    CROSSOVER_DOC: (
        "| CPU | 29.473 s | 1.000x | reference |",
        "| forced CUDA baseline | 139.674 s | 4.739x slower | 4,591 numerical metrics pass |",
        "| `auto` | 29.172 s | 0.990x of CPU sample | selected CPU; 4,591 metrics pass |",
        "The current policy uses rank-local cell count and conservative host/device\n"
        "facts. It does not know a future stop condition or timestep count, model DFT\n"
        "monitor/frequency launch count",
        "are structural v1.1.0 candidates and require a separately\n"
        "approved performance matrix.",
    ),
}


def doc_contract_matches(relative: str, text: str) -> bool:
    return all(fragment in text for fragment in DOC_CONTRACTS[relative])


def _normalize_markdown_prose(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _replace_markdown_prose_fragment(
    text: str, fragment: str, replacement: str = ""
) -> str:
    pattern = r"\s+".join(re.escape(part) for part in fragment.split())
    return re.sub(pattern, replacement, text, count=1)


def _extract_readme_architecture_section(text: str) -> str | None:
    matches = list(
        re.finditer(
            rf"(?m)^## {re.escape(README_ARCHITECTURE_HEADING)}\s*$",
            text,
        )
    )
    if len(matches) != 1:
        return None
    start = matches[0].end()
    following = re.search(r"(?m)^## [^#].*$", text[start:])
    end = start + following.start() if following else len(text)
    return text[start:end]


def _split_readme_architecture_subsections(
    section: str,
) -> dict[str | None, str] | None:
    matches = list(re.finditer(r"(?m)^### ([^\n]+?)\s*$", section))
    headings = tuple(match.group(1) for match in matches)
    if headings != README_ARCHITECTURE_SUBSECTIONS:
        return None
    blocks: dict[str | None, str] = {
        None: section[: matches[0].start()] if matches else section
    }
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        blocks[match.group(1)] = section[start:end]
    return blocks


def _readme_architecture_table_row(section: str, label: str) -> str | None:
    match = re.search(
        rf"(?m)^\| {re.escape(label)} \| (.*?) \|\s*$",
        section,
    )
    return match.group(1) if match else None


def readme_gpu_architecture_contract_matches(text: str) -> bool:
    section = _extract_readme_architecture_section(text)
    if section is None:
        return False
    blocks = _split_readme_architecture_subsections(section)
    if blocks is None:
        return False
    if len(re.findall(r"[A-Za-z0-9_`./+-]+", section)) < README_ARCHITECTURE_MINIMUM_WORDS:
        return False

    for heading, required_fragments in README_ARCHITECTURE_REQUIRED_PROSE.items():
        normalized_block = _normalize_markdown_prose(blocks[heading])
        for fragment in required_fragments:
            if _normalize_markdown_prose(fragment) not in normalized_block:
                return False

    normalized_document = _normalize_markdown_prose(text).casefold()
    if any(
        _normalize_markdown_prose(overclaim).casefold() in normalized_document
        for overclaim in README_ARCHITECTURE_FORBIDDEN_OVERCLAIMS
    ):
        return False
    if any(
        re.search(pattern, normalized_document)
        for pattern in README_ARCHITECTURE_FORBIDDEN_PATTERNS
    ):
        return False

    curl_row = _readme_architecture_table_row(section, "B/D curl")
    eh_row = _readme_architecture_table_row(section, "H/E constitutive update")
    boundary_row = _readme_architecture_table_row(section, "Boundaries")
    if curl_row is None or eh_row is None or boundary_row is None:
        return False
    normalized_curl = _normalize_markdown_prose(curl_row).casefold()
    normalized_eh = _normalize_markdown_prose(eh_row).casefold()
    normalized_boundary = _normalize_markdown_prose(boundary_row).casefold()
    if not all(
        term in normalized_curl
        for term in ("pml curl auxiliaries", "conductivity", "2d beta", "bfast")
    ):
        return False
    if "anisotropic" in normalized_curl or "bloch" in normalized_curl:
        return False
    if not all(
        term in normalized_eh
        for term in (
            "diagonal/off-diagonal inverse susceptibility",
            "pml h/e auxiliaries",
            "`chi2`",
            "`chi3`",
        )
    ):
        return False
    if "conductivity" in normalized_eh or "bloch" not in normalized_boundary:
        return False

    for target in re.findall(r"\]\(([^)]+)\)", section):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0]
        if not relative or not (ROOT / relative).exists():
            return False
    return True


def active_identity_is_coherent(text_by_path: dict[str, str]) -> bool:
    if set(text_by_path) != set(ACTIVE_IDENTITY_MARKERS):
        return False
    for relative, required in ACTIVE_IDENTITY_MARKERS.items():
        if not all(fragment in text_by_path[relative] for fragment in required):
            return False
    combined = "\n".join(text_by_path[relative] for relative in text_by_path)
    return not any(value in combined for value in FORBIDDEN_ACTIVE_IDENTITIES)


class V103ReleaseContractTests(unittest.TestCase):
    def test_all_shipped_validators_use_one_version_and_anchor(self):
        self.assertEqual(VERIFY.DISTRIBUTION_VERSION, "1.0.3")
        self.assertEqual(SELF_CHECK.DISTRIBUTION_VERSION, "1.0.3")
        self.assertEqual(COMMON.GPMEEP_DISTRIBUTION_VERSION, "1.0.3")
        self.assertEqual(VERIFY.RELEASE_MANIFEST_SCHEMA_VERSION, 3)
        self.assertEqual(SELF_CHECK.RELEASE_MANIFEST_SCHEMA_VERSION, 3)
        self.assertEqual(COMMON.GPMEEP_RELEASE_MANIFEST_SCHEMA_VERSION, 3)
        anchor = MATRIX.SEALED_M3_ANCHOR
        for candidate in (
            VERIFY.SEALED_M3_ANCHOR,
            SELF_CHECK.SEALED_M3_ANCHOR,
            COMMON.GPMEEP_SEALED_M3_ANCHOR,
        ):
            self.assertTrue(COMMON.typed_json_equal(anchor, candidate))
        self.assertTrue(
            COMMON.typed_json_equal(
                VERIFY.RELEASE_MANIFEST_FIELDS,
                SELF_CHECK.RELEASE_MANIFEST_FIELDS,
            )
        )
        self.assertTrue(
            COMMON.typed_json_equal(
                VERIFY.RELEASE_MANIFEST_FIELDS,
                COMMON.GPMEEP_RELEASE_MANIFEST_FIELDS,
            )
        )

    def test_all_anchor_checks_reject_boolean_integer_type_confusion(self):
        mutated = copy.deepcopy(MATRIX.SEALED_M3_ANCHOR)
        mutated["qualification_eligible"] = 1
        self.assertFalse(
            COMMON.typed_json_equal(mutated, COMMON.GPMEEP_SEALED_M3_ANCHOR)
        )
        self.assertFalse(MATRIX._anchor_is_valid(mutated))
        manifest = copy.deepcopy(VERIFY.RELEASE_MANIFEST_FIELDS)
        manifest["source_commit"] = "a" * 40
        manifest["performance_anchor"] = mutated
        for validator in (
            VERIFY.validate_release_manifest,
            SELF_CHECK.validate_release_manifest,
            COMMON.validate_release_manifest,
        ):
            with self.subTest(validator=validator.__module__):
                with self.assertRaises(VERIFY.ReleaseManifestError):
                    validator(manifest)

    def test_package_manifest_template_matches_the_shipped_validators(self):
        text = (ROOT / "packaging/conda/release.json.in").read_text(encoding="utf-8")
        replacements = {
            "@SOURCE_COMMIT@": "b" * 40,
            "@CUDA_ARCHITECTURES@": "AUTO",
            "@CUDA_REAL_ARCHITECTURES@": "60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90",
            "@CUDA_VIRTUAL_ARCHITECTURES@": "90",
            "@RUNTIME_VALIDATED_ARCHITECTURES@": '"sm86"',
            "@CUDA_ARCHITECTURE_AUDIT_SHA256@": "b" * 64,
            "@GLIBC_AUDIT_SHA256@": "a" * 64,
        }
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        manifest = json.loads(
            text,
            object_pairs_hook=COMMON.unique_json_object,
            parse_constant=COMMON.reject_json_constant,
        )
        expected = copy.deepcopy(VERIFY.RELEASE_MANIFEST_FIELDS)
        expected["source_commit"] = "b" * 40
        self.assertTrue(COMMON.typed_json_equal(manifest, expected))

    def test_package_manifest_template_parser_rejects_duplicate_keys(self):
        text = (ROOT / "packaging/conda/release.json.in").read_text(encoding="utf-8")
        for source, replacement in {
            "@SOURCE_COMMIT@": "b" * 40,
            "@CUDA_ARCHITECTURES@": "AUTO",
            "@CUDA_REAL_ARCHITECTURES@": "60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90",
            "@CUDA_VIRTUAL_ARCHITECTURES@": "90",
            "@RUNTIME_VALIDATED_ARCHITECTURES@": '"sm86"',
            "@CUDA_ARCHITECTURE_AUDIT_SHA256@": "b" * 64,
            "@GLIBC_AUDIT_SHA256@": "a" * 64,
        }.items():
            text = text.replace(source, replacement)
        duplicated = text.replace(
            '"qualification_eligible": true',
            '"qualification_eligible": 1, "qualification_eligible": true',
        )
        with self.assertRaisesRegex(COMMON.WorkloadError, "duplicate JSON"):
            json.loads(
                duplicated,
                object_pairs_hook=COMMON.unique_json_object,
                parse_constant=COMMON.reject_json_constant,
            )

    def test_public_inputs_contain_no_superseded_means(self):
        paths = (
            ROOT / "doc/docs/GPMEEP_PERFORMANCE.md",
            ROOT / "scripts/user-workloads/run_installed_aunp_performance_matrix.py",
            ROOT / "packaging/conda/release.json.in",
        )
        forbidden = (
            "163.967196",
            "178.341156",
            "16.660751",
            "41.066427",
            "10.962155",
            "29.960866",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"{path} retains {value}")

    def test_build_and_install_docs_name_v103(self):
        builder = (ROOT / "scripts/build-conda-package.sh").read_text(encoding="utf-8")
        self.assertIn('DISTRIBUTION_VERSION="1.0.3"', builder)
        recipe = (ROOT / "packaging/conda/meta.yaml").read_text(encoding="utf-8")
        self.assertIn('{% set distribution_version = "1.0.3" %}', recipe)
        self.assertIn("{% set build_number = 0 %}", recipe)
        package_build = (ROOT / "packaging/conda/build.sh").read_text(encoding="utf-8")
        verifier = (ROOT / "scripts/verify-gpmeep-package.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('CONDA_BUILD_NUMBER = 0', verifier)
        self.assertIn(
            'CONDA_BUILD_STRING = "cuda124_mpi_openmpi_py311_0"', verifier
        )
        packaged_docs = tuple(
            re.findall(
                r'"\$\{SRC_DIR\}/doc/docs/(GPMEEP_[A-Z0-9_]+\.md)"',
                package_build,
            )
        )
        self.assertEqual(packaged_docs, PACKAGED_DOCS)
        self.assertIn('"${PREFIX}/share/gpmeep/docs/"', package_build)
        for relative in ("README.md", "doc/docs/GPMEEP_INSTALLATION.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("gpmeep-1.0.3-", text)
            self.assertNotIn("gpmeep-1.0.1-cuda124_mpi_openmpi_py311_1", text)
            self.assertIn("py311_0.conda", text)

    def test_active_release_identity_is_coherent(self):
        text_by_path = {
            relative: (ROOT / relative).read_text(encoding="utf-8")
            for relative in ACTIVE_IDENTITY_MARKERS
        }
        self.assertTrue(active_identity_is_coherent(text_by_path))

    def test_active_identity_rejects_every_stale_build_class(self):
        text_by_path = {
            relative: (ROOT / relative).read_text(encoding="utf-8")
            for relative in ACTIVE_IDENTITY_MARKERS
        }
        target = "scripts/build-conda-package.sh"
        for forbidden in FORBIDDEN_ACTIVE_IDENTITIES:
            with self.subTest(forbidden=forbidden):
                mutated = dict(text_by_path)
                mutated[target] += f"\n{forbidden}\n"
                self.assertFalse(active_identity_is_coherent(mutated))

    def test_public_docs_publish_exact_labeled_rows_ratios_and_identities(self):
        for relative in DOC_CONTRACTS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertTrue(doc_contract_matches(relative, text), relative)

    def test_public_doc_contract_rejects_swapped_rows_and_hashes(self):
        report_hash = MATRIX.SEALED_M3_ANCHOR["report_sha256"]
        complete_hash = MATRIX.SEALED_M3_ANCHOR["complete_sha256"]
        for relative in (PERFORMANCE_DOC, HISTORICAL_RELEASE_NOTES_DOC):
            text = (ROOT / relative).read_text(encoding="utf-8")
            swapped_rows = (
                text.replace("| GPU1 |", "| __GPU__ |")
                .replace("| GPU2 |", "| GPU1 |")
                .replace("| __GPU__ |", "| GPU2 |")
            )
            self.assertFalse(doc_contract_matches(relative, swapped_rows), relative)
            swapped_hashes = (
                text.replace(report_hash, "__REPORT_HASH__")
                .replace(complete_hash, report_hash)
                .replace("__REPORT_HASH__", complete_hash)
            )
            self.assertFalse(doc_contract_matches(relative, swapped_hashes), relative)

    def test_current_docs_reject_identity_and_crossover_timing_mutations(self):
        notes = (ROOT / CURRENT_RELEASE_NOTES_DOC).read_text(encoding="utf-8")
        changed_identity = notes.replace(
            "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_0.conda",
            "gpmeep-1.0.3-cuda124_mpi_openmpi_py311_1.conda",
        )
        self.assertFalse(
            doc_contract_matches(CURRENT_RELEASE_NOTES_DOC, changed_identity)
        )

        crossover = (ROOT / CROSSOVER_DOC).read_text(encoding="utf-8")
        changed_timing = crossover.replace(
            "| forced CUDA baseline | 139.674 s |",
            "| forced CUDA baseline | 139.000 s |",
        )
        self.assertFalse(doc_contract_matches(CROSSOVER_DOC, changed_timing))

    def test_current_notes_reject_scope_and_evidence_limit_mutations(self):
        notes = (ROOT / CURRENT_RELEASE_NOTES_DOC).read_text(encoding="utf-8")
        mutations = (
            (
                "It does not\nchange the accepted CUDA FDTD numerical kernels or their FP32 policy.",
                "It does\nchange the accepted CUDA FDTD numerical kernels and their FP32 policy.",
            ),
            (
                "The requested policy must exactly match the\n  "
                "recorded native and PTX inventories.",
                "The requested policy need not match the\n  "
                "recorded native and PTX inventories.",
            ),
            (
                "These measurements are production observations, not\n"
                "portable v1.0.3 performance promises.",
                "These measurements are portable v1.0.3 performance promises.",
            ),
            (
                "v1.0.3 includes no speculative kernel\n"
                "or MPI transport rewrite",
                "v1.0.3 includes a speculative kernel\n"
                "and MPI transport rewrite",
            ),
        )
        for source, replacement in mutations:
            with self.subTest(source=source):
                mutated = notes.replace(source, replacement)
                self.assertNotEqual(mutated, notes)
                self.assertFalse(
                    doc_contract_matches(CURRENT_RELEASE_NOTES_DOC, mutated)
                )

    def test_readme_gpu_architecture_contract_matches(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        self.assertTrue(readme_gpu_architecture_contract_matches(readme))

    def test_readme_gpu_architecture_requires_each_caveat_in_its_subsection(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        for heading, fragments in README_ARCHITECTURE_REQUIRED_PROSE.items():
            for fragment in fragments:
                with self.subTest(heading=heading, fragment=fragment):
                    mutated = _replace_markdown_prose_fragment(readme, fragment)
                    self.assertNotEqual(mutated, readme)
                    self.assertFalse(
                        readme_gpu_architecture_contract_matches(mutated)
                    )

    def test_readme_gpu_architecture_rejects_appended_overclaims(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        next_heading = "\n## Numerical and environment migration notes"
        self.assertIn(next_heading, readme)
        for overclaim in README_ARCHITECTURE_FORBIDDEN_OVERCLAIMS:
            with self.subTest(overclaim=overclaim):
                mutated = readme.replace(
                    next_heading,
                    f"\n\n{overclaim}\n{next_heading}",
                    1,
                )
                self.assertTrue(
                    "it is not a guarantee that the selected backend is faster"
                    in _normalize_markdown_prose(mutated)
                )
                self.assertFalse(readme_gpu_architecture_contract_matches(mutated))
                self.assertFalse(
                    readme_gpu_architecture_contract_matches(
                        f"{readme.rstrip()}\n\n{overclaim}\n"
                    )
                )

    def test_readme_gpu_architecture_rejects_fragments_outside_or_without_prose(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        fragment = "CPU is the default"
        moved = _replace_markdown_prose_fragment(readme, fragment)
        moved += f"\n\n{fragment}\n"
        self.assertFalse(readme_gpu_architecture_contract_matches(moved))

        fragments_only = [f"## {README_ARCHITECTURE_HEADING}"]
        fragments_only.extend(README_ARCHITECTURE_REQUIRED_PROSE[None])
        for heading in README_ARCHITECTURE_SUBSECTIONS:
            fragments_only.append(f"### {heading}")
            fragments_only.extend(README_ARCHITECTURE_REQUIRED_PROSE[heading])
        self.assertFalse(
            readme_gpu_architecture_contract_matches("\n".join(fragments_only))
        )

    def test_readme_gpu_architecture_rejects_duplicate_or_reordered_sections(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        section = _extract_readme_architecture_section(readme)
        self.assertIsNotNone(section)
        marker = "\n## Numerical and environment migration notes"
        duplicate = readme.replace(
            marker,
            f"\n## {README_ARCHITECTURE_HEADING}{section}{marker}",
            1,
        )
        self.assertFalse(readme_gpu_architecture_contract_matches(duplicate))

        first, second = README_ARCHITECTURE_SUBSECTIONS[:2]
        reordered = (
            readme.replace(f"### {first}", "### __FIRST__", 1)
            .replace(f"### {second}", f"### {first}", 1)
            .replace("### __FIRST__", f"### {second}", 1)
        )
        self.assertFalse(readme_gpu_architecture_contract_matches(reordered))

    def test_readme_gpu_architecture_allows_reflow_and_equivalent_determiner_edit(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        reflowed = readme.replace(
            "The acceleration layer sits below the existing\n`fields::step()` pipeline",
            "The acceleration layer sits below the existing `fields::step()` pipeline",
            1,
        )
        reflowed = reflowed.replace(
            "both its CPU and CUDA lanes",
            "both CPU and CUDA lanes",
            1,
        )
        self.assertNotEqual(reflowed, readme)
        self.assertTrue(readme_gpu_architecture_contract_matches(reflowed))

    def test_readme_gpu_architecture_locks_phase_attribution_and_local_links(self):
        readme = (ROOT / README_DOC).read_text(encoding="utf-8")
        bad_curl = readme.replace(
            "PML curl auxiliaries, electric/magnetic conductivity",
            "PML curl auxiliaries, anisotropic coefficients, Bloch phase, electric/magnetic conductivity",
            1,
        )
        self.assertFalse(readme_gpu_architecture_contract_matches(bad_curl))

        bad_eh = readme.replace(
            "PML H/E auxiliaries, integrated-source terms",
            "PML H/E auxiliaries, conductivity, integrated-source terms",
            1,
        )
        self.assertFalse(readme_gpu_architecture_contract_matches(bad_eh))

        missing_link = readme.replace(
            "cuda/src/runtime.cu", "cuda/src/does-not-exist.cu", 1
        )
        self.assertFalse(readme_gpu_architecture_contract_matches(missing_link))


if __name__ == "__main__":
    unittest.main()
