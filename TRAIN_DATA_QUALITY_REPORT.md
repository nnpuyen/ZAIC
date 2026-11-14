# Training Data Quality and Pipeline Assessment

## Scope and method

This report covers every supported source under `train\train\data`:

- 30 business datasets
- 128 files: 107 `.xlsx`, 20 `.csv`, and 1 `.sqlite`
- 339 workbook sheets / database tables
- 3,997,277 data rows and 27,191,471 cells

The scan used streaming reads for Excel files, CSV parsing with the same
encoding order as the project loader, and SQLite table reads. Empty strings and
nulls were treated as missing. Duplicate counts are exact repeated complete
rows, not inferred business-key duplicates. Date validation was structural
(common date forms, Excel serials, and native Excel dates); semantic date
validation still needs business-key and calendar checks.

## Executive summary

| Finding | Quantity | Percentage | Risk |
|---|---:|---:|---|
| Files successfully read in the project environment | 128 / 128 | 100% | Low |
| Tables containing at least one missing cell | 230 / 339 | 67.85% | Medium to high |
| Missing cells overall | 741,338 / 27,191,471 | 2.73% | Medium |
| Tables with exact duplicate rows | 80 / 339 | 23.60% | Medium |
| Repeated rows overall | 104,316 / 3,997,277 | 2.61% | Medium |
| Empty tables | 0 / 339 | 0% | Low |
| Structurally suspicious date values | 0 / 113,491 populated date-hint values | 0% observed | Medium: semantic validation is incomplete |
| CSVs read with the first `utf-8-sig` attempt | 20 / 20 | 100% | Low |

The most important issue is not the overall missing-cell rate; it is the
concentration of missingness in particular tables. For example, several
`biz_103` monthly detail sheets have roughly 59–62% missing cells, and some
small product/reference tables have more than 50% missing cells. These tables
need column-level business rules before aggregation.

## Issues requiring resolution

### 1. Missing values are widespread and not semantically handled

- **67.85% of tables** contain missing cells.
- **2.73% of all cells** are missing.
- The worst observed table is `biz_124/SoTay_ChiPhi.xlsx`, with about 96.1%
  missing cells.
- Other high-risk examples include `biz_103/DonHang_ChiTiet.xlsx` monthly
  sheets (about 59–62% missing) and product/reference tables in `biz_108`,
  `biz_121`, and `biz_122` (about 53–61% missing).

**Why this matters:** a null can mean “not applicable”, “unknown”, “not yet
paid”, or an export defect. Treating all nulls as equivalent can distort
revenue, debt, inventory, and ranking answers.

**Current pipeline behavior:** the loader keeps nulls; the profiler reports
`null_ratio` per column. No imputation, required-column policy, null semantic
classification, or query-time guard is applied.

**Resolution coverage:** 100% of loaded tables are profiled for null ratios,
but 0% of missing values are automatically resolved. This is intentional
preservation, not data cleaning.

### 2. Exact duplicate rows need business-key classification

- **80 tables** contain exact duplicate rows.
- There are **104,316 repeated row instances**.
- The largest concentration is in `biz_103/DonHang_ChiTiet.xlsx`, where the
  monthly sheets contain repeated rows.

The repeated rows may be valid repeated transactions or export duplication.
Removing them globally would be unsafe. The current loader only removes exact
duplicates when a table name contains a narrow order-key phrase and does not
look like an event table. The scan found no clear order-key match among these
80 duplicate tables using the same normalized phrase family, so the current
heuristic automatically resolves **0 / 80 duplicate tables (0%)**.

**Recommendation:** define a per-source business key (for example order ID +
line ID, invoice ID + product ID, or transaction ID), then classify:

1. exact duplicate export rows;
2. valid repeated event rows;
3. conflicting rows sharing a business key.

Only category 1 should be removed, and the removal count must be recorded in
evidence.

### 3. Date values pass a basic structural scan, but semantic validation is missing

There are **113,491 populated values in columns that look date-like**. No value
was structurally suspicious in the scan. This does **not** prove date
correctness: it does not validate impossible dates, timezone semantics,
date-vs-datetime consistency, future-record policy, or Vietnamese lunar-calendar
mapping.

The loader attempts Excel serial dates and mixed textual dates. If any
non-null value in a candidate date column fails parsing, the entire column is
left in its original representation. That fail-open behavior can later cause
inconsistent comparisons.

**Resolution coverage:** 100% of date-hint columns are eligible for the
loader's inference attempt; 0 structural failures were observed. Semantic
validation coverage is not measured and should be treated as unknown, not 100%.

### 4. Schema and relationship integrity is not checked

The scan found no empty tables, but the pipeline does not validate:

- required columns by source type;
- primary-key uniqueness;
- foreign-key coverage between orders, order lines, products, customers,
  payments, and inventory;
- orphan order lines;
- inconsistent units or product codes;
- conflicting duplicate business keys;
- numeric currency fields containing non-numeric text.

This is the largest unmeasured risk because a table can be non-empty and
parseable while still producing an incorrect answer.

**Resolution coverage:** 0% measured. The current profiler reports row counts,
null ratios, distinct counts, and samples, but not relational integrity.

### 5. Source naming and source conventions are heterogeneous

The dataset mixes Vietnamese and English naming, monthly-sheet exports,
multiple file layouts, CSV, Excel, and SQLite. This is supported at ingestion,
but there is no canonical source-to-domain schema. The same business concept
can therefore be represented by different column names and different
transaction grain.

**Resolution coverage:** file-format coverage is 100% for the observed
extensions. Semantic schema normalization coverage is 0%.

## What the current pipeline resolves

| Pipeline capability | Evidence in code | Observed coverage | Assessment |
|---|---|---:|---|
| Supported CSV, Excel, and SQLite ingestion | `datamate/ingestion/loader.py` | 128 / 128 files, 100% | Works in the project virtual environment |
| CSV encoding fallback | `loader.py` `_read_csv` | 20 / 20 CSVs, 100% used `utf-8-sig` | Good compatibility, but encoding should be recorded in evidence |
| Duplicate-column-name removal | `loader.py` `register` | Not separately measured | Prevents duplicate column identifiers, but may discard a source column |
| Exact duplicate removal | `loader.py` `_deduplicate_export_rows` | 0% of the 80 duplicate tables matched the narrow key heuristic | Too narrow for this dataset |
| Date type inference | `loader.py` `_parse_date_series` / `_infer_column_type` | 0 structural failures observed | Needs explicit validation and failure reporting |
| Numeric text normalization | `loader.py` `_infer_column_type` | Not separately measured | Handles only a narrow comma/decimal pattern |
| Null detection and profiling | `datamate/ingestion/profiler.py` | 339 / 339 tables, 100% | Detection works; resolution does not |
| README, rulebook, and calendar context | `datamate/context.py` | Context is wired into the pipeline | Useful, but missing-source warnings are not explicit |
| Identifier and filter validation | `datamate/execution.py` | Applied to executed plans | Good fail-fast behavior for several filter types |
| Evidence and error output | `datamate/pipeline.py` | Per-question evidence is written | Strong reproducibility feature |

Percentages above distinguish **detection**, **transformation**, and
**resolution**. Profiling a missing value is not the same as resolving it.

## Pipeline issues to fix

1. **Replace filename/table-name duplicate heuristics with source manifests.**
   A manifest should declare table grain, business key, event-table status,
   required columns, and duplicate policy.
2. **Add a data-quality gate after ingestion.** Emit per-table checks for
   required columns, null thresholds, key uniqueness, orphan keys, numeric
   coercion failures, and date ranges. Fail or warn explicitly rather than
   silently accepting questionable data.
3. **Make date parsing auditable.** Store original value, parsed value, parser
   path, failure count, min/max date, and future-date count. Do not silently
   retain an entire unparsed date column after one failure.
4. **Add column semantic policies for nulls.** For example, distinguish
   missing customer ID (walk-in customer) from missing payment date (unpaid or
   malformed payment record). Require explicit aggregation behavior for
   nullable measure columns.
5. **Normalize source schemas before planning.** Map equivalent concepts such as
   order ID, invoice ID, product code, quantity, amount, order date, and status
   into canonical names while retaining the original source column.
6. **Validate relationships before answering questions.** Especially check
   order-to-line, product-to-line, customer-to-order, payment-to-order, and
   inventory-to-product relationships.
7. **Expand execution validation to match the declared schema operators.**
   `planning/schemas.py` declares operators such as `not_contains`,
   `starts_with`, `regex`, and others, while `execution.py` implements only a
   subset. Unsupported valid plans can fail at runtime instead of receiving a
   clear validation error.
8. **Record data-quality metrics in question evidence.** Each answer should
   include the relevant table's null rate, duplicate policy, date validation
   status, and any excluded rows so an answer can be audited.
9. **Add representative fixtures and tests for the observed defects.**
   Include high-null monthly sheets, repeated event rows, missing keys,
   malformed dates, mixed numeric formats, and source-specific Vietnamese
   column names.

## Suggested priority

### P0 — before trusting production answers

- Introduce source manifests and business-key duplicate policies.
- Add required-column, numeric, date-range, and referential-integrity gates.
- Make unresolved data-quality warnings visible in the answer evidence.

### P1 — improve answer correctness

- Canonicalize source schemas and units.
- Add explicit null semantics for revenue, payment, debt, inventory, and
  customer fields.
- Expand query-plan operator validation/compilation consistently.

### P2 — improve maintainability and performance

- Cache profiling results per file fingerprint.
- Avoid rescanning all columns for every question.
- Store compact profiling summaries instead of repeatedly rebuilding them.
- Add dashboards for null, duplicate, date, and relationship metrics by
  business dataset.

## Bottom line

The ingestion path is operationally strong for this training set: all 128
files were readable, no table was empty, and the existing profiler can report
null ratios for every loaded table. However, the pipeline currently **detects
missingness rather than resolving it**, has **no measured relational-integrity
coverage**, and automatically handles only **0% of the 80 duplicate-table
cases** found by the scan. The next correctness milestone should be a
source-manifest-driven quality gate with explicit business keys, null
semantics, date validation, and relationship checks.
