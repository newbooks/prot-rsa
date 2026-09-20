# Residue SASA and Dual Contextual Exposure Fractions

**Status:** Ready for implementation.

## Purpose and scope

Produce one residue-level report containing contextual exposure for both the
complete residue (`ALL`) and its side-chain atom subset (`SIDE`). Both values
use the observed residue conformation as the reference, rather than a fixed
residue-type or maximum-ASA table.

For residue `r` and atom subsets `A_ALL(r)` and `A_SIDE(r)`:

```text
sasa_all[r]       = sum(protein_atom_sasa[a] for a in A_ALL(r))
sasa_all_ref[r]   = SASA of A_ALL(r) when residue r is isolated
sasa_all_ratio[r] = sasa_all[r] / sasa_all_ref[r]

sasa_side[r]       = sum(protein_atom_sasa[a] for a in A_SIDE(r))
sasa_side_ref[r]   = SASA of A_SIDE(r) when the side-chain subset is isolated
sasa_side_ratio[r] = sasa_side[r] / sasa_side_ref[r]
```

The protein numerator is calculated in the complete normalized structure. Each
reference calculation contains only the selected subset from the target
residue; atoms from other residues and atoms excluded from that subset are
not present as occluders. This makes the ALL and SIDE denominators consistent
with their respective numerators and measures exposure relative to the same
conformation without surrounding residues.

The existing atom-SASA report remains unchanged. This specification changes
only the residue report and removes the need for a CLI `ALL`/`SIDE` selection
for residue output: both metrics are emitted in every successful run.

## Atom subsets

`A_ALL(r)` contains every normalized atom belonging to residue `r`, subject to
the existing hydrogen and hetero-atom options.

`A_SIDE(r)` contains the normalized atoms in `A_ALL(r)` whose atom names are
not in the canonical backbone set:

```text
N, CA, C, O, OXT
```

Names are compared after the existing atom-name normalization. `OXT` is
classified as backbone so terminal residues receive the same treatment as
internal residues. No inferred atoms, residue templates, bond perception, or
alternate-location data may be introduced at this stage.

The implementation must preserve the existing normalization behavior:

- `--use-h` controls whether explicit hydrogen records are retained;
- radius assignment uses the existing ProtOr/explicit tables;
- loose hetero-component filtering uses `--preserve-het`; and
- the same probe radius, sphere points, backend, and worker settings are used
  for protein and reference calculations.

## Output file and schema

The existing output path rules remain authoritative:

```text
1LYZ.pdb    -> 1LYZ.res.sas
1LYZ.cif.gz -> 1LYZ.res.sas
```

After the explanatory comment lines, the file must contain exactly this
tab-separated header row:

```text
res	chain	seq	ins	all	a_ref	a_cef	side	s_ref	s_cef
```

Each column must have one leading `#` comment line matching the descriptions
in `RESIDUE_CEF_COMMENTS`. Parsers must skip these comment lines before reading
the header.

The first four fields are copied from the first normalized atom for each
residue. The residue identity key remains:

```text
(residue_name, chain_id, residue_sequence, insertion_code)
```

Rows are emitted in first-seen residue order. Sequence identifiers remain
strings, including nonnumeric mmCIF identifiers. The file is UTF-8 TSV with
comment lines, one header row, and a trailing newline. Numeric values are formatted with exactly three
digits after the decimal point; intermediate values are never rounded. For
residues without retained SIDE atoms, the three SIDE fields contain the
literal `NA`.

## Calculation algorithm

1. Read the structure and resolve models/alternate locations using the
   existing input contract.
2. Normalize atoms once and calculate complete-structure atom SASA once.
3. Group normalized atom indices by the four-field residue key while
   preserving first-seen order.
4. For each residue, construct an ALL coordinate/radius view and a SIDE view
   by filtering out canonical backbone atom names.
5. Calculate the isolated ALL reference SASA using only the ALL view.
6. Calculate the isolated SIDE reference SASA using only the SIDE view.
7. Sum complete-structure atom SASAs over the ALL and SIDE index sets.
8. Compute both ratios and validate their numerical bounds.
9. Write one row containing both metric triplets only after all calculations
   succeed. A failure must not leave a partial residue file.

The first implementation may call the existing validated spatial function once
per reference view. It must not invoke the CLI parser or write temporary files.
The same validated sphere-point array must be passed to all reference calls.

## Numerical behavior

For each nonempty subset, require:

```text
0 <= sasa_all <= sasa_all_ref
0 <= sasa_side <= sasa_side_ref
0 <= sasa_all_ratio <= 1
0 <= sasa_side_ratio <= 1
```

Allow `1e-12` floating-point slack at the bounds and clamp values within that
slack before serialization. Larger violations, nonfinite values, or a
nonpositive reference area must raise a clear error. Ratios must be computed
from unrounded sums.

## Public API and CLI

The reusable residue calculation API should return a record containing the
four identity fields and six numeric values above. Existing atom-level APIs
retain their signatures and behavior.

The CLI continues to write `.atom.sas`, `.res.sas`, and `.pqr` in one run. The
residue file always contains both ALL and SIDE columns. There is no CLI mode
switch for selecting residue columns.

## Required tests

Add focused tests for:

1. Exact comment lines, ten-column header, ordering, formatting, and output
   naming.
2. ALL aggregation and SIDE aggregation on a residue containing backbone and
   side-chain atoms.
3. Backbone exclusion for `N`, `CA`, `C`, `O`, and terminal `OXT`.
4. Isolation of ALL and SIDE references, including internal same-subset
   occlusion.
5. Grouping by all four identity fields, noncontiguous records, insertion
   codes, and nonnumeric sequence identifiers.
6. Fully exposed and partially occluded residues for both ratios.
7. Hydrogen and loose-hetero options, with atom output unchanged.
8. Serial/compiled backend agreement within the established numerical
   contract.
9. Empty SIDE subsets and invalid reference areas according to the decision
   recorded below.
10. Determinism, input nonmutation, and no partial output after failure.

## Benchmark and acceptance criteria

Benchmark complete-structure atom time, ALL reference time, SIDE reference
time, and total residue-report time for the canonical small, medium, and large
structures. Report atom and residue counts, side-chain atom counts, sphere
resolution, backend, workers, and hardware.

This stage is complete when the ten-column report is deterministic, both
numerators and references use the specified atom subsets, atom-level output is
unchanged, all numerical bounds are enforced, and the focused plus full test
suites pass.

## Resolved classification policies

### Residues with no SIDE atoms

Glycine has no conventional side-chain atom under the backbone definition
above. A residue may also have no retained SIDE atoms after filtering. Its
SIDE reference area is therefore zero and a SIDE ratio is undefined.

The three SIDE fields are serialized as `NA` for such residues and excluded
from numeric SIDE-ratio summaries. This avoids conflating “no side chain” with
zero exposed area.

### Explicit hydrogen classification

The canonical backbone names classify heavy atoms unambiguously, but explicit
hydrogen atom names can be attached to backbone or side-chain atoms and the
current normalized atom model does not provide bond connectivity.

SIDE classification is defined by the normalized atom name only. Explicit
hydrogens whose names are not in the backbone set remain in SIDE. A chemically
exact hydrogen policy would require connectivity inference or an explicit
atom-name mapping and is outside this implementation.
