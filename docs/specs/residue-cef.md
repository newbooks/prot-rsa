# Residue SASA and Contextual Exposure Fraction

**Status:** Implemented

## Purpose and scope

Add the first residue-level report on top of the validated atom-SASA
calculation. The report aggregates atom SASAs in the complete protein and
normalizes each residue by the SASA of that same residue conformation when
calculated in isolation. The normalized value is named **Contextual Exposure
Fraction (CEF)**. It is intentionally distinct from conventional RSA, which
normally uses a residue-type reference or maximum ASA.

This stage supports only `ALL` atom-selection mode. `SIDE` and `KEY` are
deferred and must remain rejected by the CLI until separate scientific
contracts are specified.

For residue `i`:

```text
sasa_inprotein[i] = sum(atom_sasa[a] for atoms a in residue i,
                        calculated in the complete structure)
sasa_reference[i] = sum(atom_sasa[a] for atoms a in residue i,
                        calculated with only residue i present)
cef[i]            = sasa_inprotein[i] / sasa_reference[i]
```

The naked-residue calculation retains the residue's observed coordinates,
selected atoms, radii, and conformation. It removes every atom belonging to a
different residue. Atoms within the residue remain occluders of one another,
so the denominator includes self-shielding caused by the residue's own
conformation. Terminal residues require no special reference treatment.

This stage does not define conventional maximum-ASA tables, residue-type
reference values, side-chain-only exposure, key-atom exposure, or residue
classification thresholds.

## Output file and schema

Every successful structure calculation writes a residue TSV alongside the atom
TSV. The existing output-path derivation is authoritative:

```text
1LYZ.pdb       -> 1LYZ.res.sas
1LYZ.cif.gz    -> 1LYZ.res.sas
```

The file must contain exactly these columns in this order:

```text
residue_name
chain_id
residue_sequence
insertion_code
sasa_inprotein
sasa_reference
cef
```

The first four fields are text copied from the normalized source atom record.
Blank chain and insertion identifiers remain blank. `residue_sequence` remains
a string; do not convert it to an integer because mmCIF identifiers and
future input formats may not be numeric.

`sasa_inprotein` and `sasa_reference` are absolute areas in Å². `cef` is a
dimensionless ratio. Numeric values must be serialized with three digits after
the decimal point. The file is UTF-8
TSV with one header row and a trailing newline. Rows are written in the order
of the first atom encountered for each residue.

The residue identity key is the four-tuple:

```text
(residue_name, chain_id, residue_sequence, insertion_code)
```

All normalized atoms with the same key belong to one output row, even if the
records are not contiguous in the input. Alternate-location resolution and
model rejection occur before grouping, using the existing structure-reader
contract. A residue key must not occur twice as separate output rows.

## Atom-selection and normalization rules

The residue report consumes the same normalized atoms used for atom SASA:

- `ALL` is the only supported mode;
- hydrogen inclusion follows `--use-h`;
- radius assignment follows the selected ProtOr or explicit radius table;
- loose hetero-component filtering follows `--preserve-het`; and
- the same probe radius and sphere-point set are used for both SASA values.

The numerator and denominator must use identical atom membership for the
residue. An atom omitted by normalization cannot contribute to either value.
The implementation must not infer missing atoms, add hydrogens, or substitute
a residue-type reference table.

## Calculation algorithm

1. Parse the structure and resolve models and alternate locations according to
   the existing input contract.
2. Normalize and filter atoms using `ALL` mode.
3. Calculate atom SASA once in complete-structure context using the selected
   production backend. Preserve the existing atom order and atom-level output.
4. Group normalized atom indices by the residue identity key while preserving
   first-seen residue order.
5. For each residue, create an isolated view containing only that residue's
   coordinates and radii. Calculate atom SASA with the same probe and sphere
   points, with no atoms from other residues available as occluders.
6. Sum the complete-structure atom SASAs into `sasa_inprotein` and the isolated
   atom SASAs into `sasa_reference`.
7. Compute `cef = sasa_inprotein / sasa_reference`.
8. Serialize one row per residue to `<base>.res.sas` only after all parsing and
   calculations succeed.

The first implementation may call the existing validated spatial function for
each isolated residue. It must not call the public parser or write temporary
files for individual residues. A later batched or compiled denominator kernel
may replace this implementation without changing the TSV contract.

The isolated calculation must use the same numerical backend requested for the
complete structure, or a documented backend-equivalence path with the same
scientific boundary rule. Backend selection must not silently change the
definition of CEF.

## Numerical behavior

With identical residue atoms and arithmetic, the complete-structure exposed
surface cannot exceed the isolated exposed surface. Require:

```text
0 <= sasa_inprotein <= sasa_reference
0 <= cef <= 1
```

Allow only floating-point roundoff at the bounds. If the computed CEF is within
`1e-12` below zero or above one, clamp that value to the corresponding bound
for serialization. A larger violation must raise a clear numerical error
rather than silently producing an invalid report. A nonpositive
`sasa_reference` is invalid for a normalized residue report and must raise a
clear error.

Do not round intermediate atom areas, residue sums, or ratios. Apply three-place
formatting only while writing the TSV. Repeated runs with identical normalized
inputs and parameters must produce identical residue rows and values.

## Public API and CLI

The existing atom-level public functions retain their signatures and behavior.
Add residue reporting through the planned residue API and CLI output path; do
not overload atom-SASA arrays with residue values. The reusable residue API
must accept normalized array data plus residue identity fields, probe settings,
sphere points, backend, and worker controls without reading files or writing
outputs.

The CLI must:

- continue to derive `.res.sas` from the input path using the existing rules;
- write the residue report in the same successful run as `.atom.sas`;
- reject `SIDE` and `KEY` with the existing not-implemented error; and
- leave no partial residue file if parsing or either SASA calculation fails.

## Required tests

Add focused `pytest` coverage for:

1. Exact header, column order, three-place numeric formatting, and output naming
   for `.pdb`, `.cif`, and compressed inputs.
2. Grouping by the complete four-field residue key, including insertion codes,
   blank chain identifiers, nonnumeric sequence identifiers, and noncontiguous
   records.
3. Atom aggregation in complete-structure context with multiple residues.
4. An isolated single-atom residue, where `sasa_reference` equals the exposed
   area of that atom without other residues.
5. Internal same-residue occlusion being retained in the denominator.
6. Terminal residues producing ordinary rows without special handling.
7. A fully exposed residue with `cef == 1` within floating-point tolerance and
   a partially occluded residue with `0 <= cef < 1`.
8. The same result under serial and supported compiled backends within the
   repository's established numerical contract.
9. `ALL` mode atom filtering, hydrogen handling, hetero filtering, and input
   nonmutation matching atom-SASA behavior.
10. Failure on zero reference area or materially invalid CEF bounds without a
    partial output file.
11. Repeated calculation determinism and preservation of atom-output results.

## Benchmark and validation requirements

Measure complete-structure atom SASA time, isolated-reference calculation time,
total residue-report time, atom count, residue count, and mean atoms per
residue for representative small, medium, and large structures. Report the
denominator cost separately because the first implementation may perform one
small isolated calculation per residue.

Compare residue outputs across serial and compiled backends and inspect CEF
distributions for values outside `[0, 1]`. Validate at least one structure
containing insertion codes, terminal residues, alternate locations, and
nonstandard retained hetero atoms when those options are enabled.

## Acceptance criteria

This stage is complete when:

- `<base>.res.sas` is written with exactly the specified seven columns;
- `sasa_inprotein` is the sum of complete-structure atom SASAs;
- `sasa_reference` uses the same residue conformation in isolation;
- CEF is calculated only for `ALL` mode and is numerically bounded;
- residue identity, ordering, and formatting are deterministic;
- atom output and public atom-SASA behavior remain unchanged;
- focused and full test suites pass; and
- benchmark conditions and denominator cost are documented.
