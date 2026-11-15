# Current DataMate Pipeline

This is the short, practical description of the pipeline currently implemented
in this repository.

## What it does

DataMate takes business files and natural-language questions, turns each
question into a safe query, runs it in DuckDB, and writes the answer plus the
evidence used to produce it.

```text
Business files + business documents
              |
              v
        Load and normalize
              |
              v
       Profile tables/schema
              |
              v
       Build question context
              |
              v
  Retrieve only relevant context
              |
              v
      Create a QueryPlan
   (LLM planner or heuristic)
              |
              v
 Complete and validate the plan
              |
              v
    Compile parameterized SQL
              |
              v
        Run in DuckDB
              |
              v
       Answer + evidence files
```

## Step by step

1. **Load data**

   `datamate/ingestion/loader.py` reads CSV, Excel, and SQLite files from one
   business directory. It creates a `BusinessDatabase` containing DuckDB
   tables and metadata.

   During loading, the pipeline repairs common encoding problems, cleans table
   and column names, applies an optional `canonical_mapping.yml`, infers dates
   and numbers, and removes exact duplicate rows from order-style exports.
   Source and column-mapping information is kept for traceability.

2. **Profile the data**

   `datamate/ingestion/profiler.py` records the tables, columns, types, row
   counts, null ratios, distinct values, samples, and numeric ranges. Raw rows
   are not sent to the planner.

3. **Build context**

   `datamate/context.py` combines the schema with:

   - `README*` files in the business directory;
   - `rulebook.md` when available;
   - `calendar.csv` when available.

4. **Retrieve relevant context**

   `datamate/planning/retrieval.py` keeps every table and column in model
   context, selects relevant rulebook sections, and resolves date or
   lunar-calendar information needed by the question.

5. **Create a query plan**

   The selected planner returns a `QueryPlan` describing the table, metrics,
   filters, joins, grouping, ordering, and answer type.

   `datamate/planning/plan_extraction.py` accepts common response envelopes and
   formatting, extracts the plan, and validates it against the `QueryPlan`
   schema. Conflicting or incomplete candidates are rejected rather than
   guessed.

   - `--planner llm` uses an OpenAI-compatible endpoint. Supported provider
     presets are `local`, `openrouter`, `huggingface`, `groq`, and `gemini`.
   - `--planner heuristic` uses the deterministic offline planner.

6. **Complete and validate the plan**

   `datamate/planning/completion.py` fills in only high-confidence details
   grounded in the loaded metadata. For example, it can resolve canonical
   table/column aliases, add a date range or `as_of_date` filter, add a safe
   join, or choose a uniquely identified measure.

   Invalid or ambiguous plans are rejected. The execution layer then checks
   that tables, columns, joins, filters, and limits are supported.

7. **Compile and execute SQL**

   `datamate/execution.py` normalizes column names, quotes identifiers, binds
   values as parameters, compiles the plan to DuckDB SQL, and executes it.
   Comparison queries are run when the plan contains `comparison_filters`.

8. **Write the result**

   Answered questions are written with `id`, `type`, and `answer`; `type` comes
   from the query plan. Failed questions are written with `id`, `type`, and a
   diagnostic `reason`; non-finite scalar results and empty result sets are
   surfaced as `abstain` instead of being serialized as nulls or empty answers.
   A failure for one question does not stop the remaining questions.
   `list` and `table` answers are collections. `number`, `percent`, `string`,
   `date`, and `boolean` answers are always one value; plans that cannot
   produce that shape are rejected.
   Database connections are closed after processing.

## Main entry points

```powershell
# Create a schema/context document
python .\profile_business.py .\public_test\public_test\data\dev_001 -o .\schema.json

# Run with the default LLM planner
python .\run_questions.py .\questions_samples.jsonl .\public_test\public_test\data --planner llm

# Run offline with the heuristic planner
python .\run_questions.py .\questions_samples.jsonl .\public_test\public_test\data --planner heuristic
```

The public Python entry point is:

```python
from datamate.pipeline import answer_questions
```

## Output files

When the CLI receives an output directory, results are written under:

```text
outputs/
└── <question-file>_answers/
    ├── answers.jsonl
    ├── token_usage.jsonl
    └── <question-id>/
        ├── answer.json
        └── evidence/
            ├── 01_question.json
            ├── 02_loaded_data.json
            ├── 03_context.json
            ├── 04_planner_response.txt       (LLM runs, when available)
            ├── 05_planner_plan.json          (when available)
            ├── 06_validated_plan.json
            ├── 07_query_trace.json
            ├── 07_query_*_*.sql
            ├── 08_query_result.json
            └── 09_error.json                  (only when unanswered)
```

Optional `--schema-output`, `--trace-output`, `--rulebook`, `--calendar`, and
`--evidence-dir` arguments can change where supporting files are written or
which business documents are used.

## Important boundaries

- Pydantic checks the QueryPlan structure. The local table/column reference
  validator is disabled by default; generated SQL is checked by DuckDB when it
  executes.
- The pipeline uses only schema/context information for planning, not raw table
  rows in the prompt.
- Plans must still reference CTE output aliases through a CTE in the `FROM` or
  `JOIN` list; defining a CTE alone does not make its columns available.
- Evidence files are the best place to debug an answer: inspect the retrieved
  context, validated plan, SQL trace, and query result in that order.
