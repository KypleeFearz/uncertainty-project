# extension_apertus: Apertus als drittes Modell (post-hoc)

Nachträgliche Erweiterung nach dem Freeze (Protokoll 3942fc49…), auf Wunsch der Betreuung (Feedback 30.10.2026).
Nicht im vorab festgelegten Analyseplan, explorativ. Der eingefrorene Code wird nur gelesen.

| Datei / Ordner | Inhalt |
|---|---|
| `run_apertus.py` | Sammelt die Antworten (Public AI, `swiss-ai/apertus-v1.5-70b`, Temperatur 0, gleiche 5 Prompts) |
| `outputs_v2/` | Eine JSON-Datei pro Fall und Prompt, `*_failed.jsonl` = ungültige Antworten, Lauf 7.11.2026 |
| `archiv_v1/` | Fehlerlog des verworfenen ersten Laufs (mit json_schema, vom Anbieter mitten im Lauf gesperrt) |
| `evaluate_apertus.py` | Auswertung mit den eingefrorenen Funktionen: Rohvergleich, Transfer, Fehlerprofil, ungültige Antworten |
| `results/` | Ergebnis-CSVs pro Teil (`test` = MNLI, `shift` = SNLI), `apertus_vergleich.png`, `run_info.json` |
| `test_apertus.py` | Tests: Ausschlussregel, Prompt-Fingerprint, Reproduktion Notebook 03, Siegel |
| `TEXTE_BERICHT_FOLIE_MAIL.md` | Berichtsabschnitt, Folie, Mail an Thomas |

Ausführen (im Projektordner):

    python extension_apertus/evaluate_apertus.py     # ca. 5 Min., schreibt nur nach results/
    python -m pytest extension_apertus -q            # Tests
    python tools/verify_freeze.py                    # Siegel prüfen

Oder Notebook `notebooks/05_apertus.ipynb` von oben nach unten ausführen.

Regeln: ein Fall braucht alle 5 Prompts (sonst ausgeschlossen, wie im eingefrorenen Code). Keine Bäume oder Schwellen
werden auf Apertus angepasst. Vergleiche laufen auf den gemeinsamen Fällen (310 MNLI, 294 SNLI).
