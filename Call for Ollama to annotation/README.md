# Ollama Q4 Amount Annotation

## Files
- `run_ollama_amount.py`: batch inference entry for amount extraction.
- `config_amount.py`: default input/output/model settings.
- `prompt_utils.py`: prompt builder aligned with SFT amount task format.
- `parser_utils.py`: robust JSON parsing and output normalization.
- `io_utils.py`: input iteration and output write helpers.

## Output Layout
Outputs are created in:

`data/ollama annotation/<YYYYMMDD_HHMMSS>/<YEAR>/<YEAR-MONTH>.json`

Each record keeps only:
- `source_file`
- `case_no`
- `target_amount`
- `amount_type`
- `source_zone`

## Smoke Test Example
```powershell
python "Call for Ollama to annotation\run_ollama_amount.py" `
  --input-dir "data\yishen_partition" `
  --output-base-dir "data\ollama annotation" `
  --limit 3 `
  --workers 1 `
  --debug
```
