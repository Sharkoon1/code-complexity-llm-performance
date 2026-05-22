# Projektarbeit: Codekomplexität & LLM-Performance

**Autor:** Mirco Böddecker
**Betreuer:** Prof. Dr. rer. nat. habil. Matthias Dehmer

Empirische Analyse: Wie korrelieren verschiedene Codekomplexitätsmaße mit dem Erfolg von Large Language Models bei Codeänderungen? Datengrundlage: SWE-Bench Verified.

---

## Arbeitsschritte

### Schritt 1 — Setup

- Python 3.11+ installieren
- Projekt initialisieren: `uv init` (kein conda nötig)
- Basis-Dependencies:
  ```
  uv add pandas datasets radon tree-sitter tree-sitter-python tiktoken
  uv add unidiff gitpython scipy statsmodels seaborn matplotlib jupyter
  ```
- LM-CC-spezifische Dependencies (lokales LLM für Token-Entropie):
  ```
  uv add torch transformers accelerate
  ```
- Ordnerstruktur anlegen:
  ```
  data/raw/         # Roh-Datensatz, Predictions
  data/interim/     # Zwischenstände
  data/entropies/   # Gecachte Token-Entropien pro Snippet
  data/processed/   # Finaler Datensatz
  src/              # Wiederverwendbare Module
  notebooks/        # Exploration & Analyse
  results/          # Plots, Tabellen
  ```
- Git-Repo + `.gitignore` (data/, __pycache__/, .venv/)
- HuggingFace-Account einrichten, Token erstellen (für CodeLlama-Zugang)

### Schritt 2 — SWE-Bench-Daten laden

- Datensatz via HuggingFace laden:
  ```python
  from datasets import load_dataset
  ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
  df = ds.to_pandas()
  df.to_parquet("data/raw/swebench_verified.parquet")
  ```
- Wichtige Felder verstehen: `instance_id`, `repo`, `base_commit`, `patch`, `test_patch`, `problem_statement`
- In Notebook explorieren: Wie viele Tasks? Welche Repos? Patch-Größen-Verteilung?

### Schritt 3 — DeepSeek-V3 Predictions vom Leaderboard holen

- SWE-Bench GitHub-Repo (`swebench/SWE-bench`): Pfad `experiments/evaluation/verified/`
- DeepSeek-V3-Submission auswählen (Anlehnung an Xie et al. 2026)
- Falls mehrere DeepSeek-Submissions: möglichst die mit minimalem Agent-Framework wählen,
  um Modell-Performance nicht durch Tooling zu verfälschen
- Auswahl im Methodik-Kapitel begründen + Submission-Datum/Hash notieren
- `results.json` mit den resolved/unresolved-IDs nach `data/raw/predictions/` laden
- Skript `src/load_predictions.py` baut daraus eine Tabelle:
  `instance_id → resolved: bool`
- Speichern: `data/raw/predictions/deepseek_v3_labels.parquet`


### Schritt 4 — Patches parsen

- Mit `unidiff` aus dem `patch`-Feld extrahieren:
  - Geänderte Dateien
  - Geänderte Zeilenbereiche
- Filter anwenden (begründen im Methodik-Kapitel!):
  - Nur Single-File-Patches
  - Max. 3 geänderte Funktionen pro Task
  - Optional: Patch-Größe begrenzen
- Speichern: `data/interim/patches_parsed.parquet`

### Schritt 5 — Pre-Patch-Code extrahieren

- Pro Task: betroffene Datei am `base_commit` holen
  - Empfohlen: GitHub-API + lokales Caching (schneller als Repo-Clone)
  - GitHub-Token einrichten (Rate Limits!)
- AST-Parsing der Datei: enthaltende Funktion jeder Änderung herausschneiden
- Speichern: `data/interim/pre_patch_code.parquet`
- **Sanity-Check:** 20 Tasks manuell prüfen — ist der Code vollständig & korrekt?

### Schritt 6 — Klassische & strukturelle Komplexitätsmaße

Pro extrahierter Funktion:

| Maß | Tool / Vorgehen |
|---|---|
| Cyclomatic Complexity | `radon.complexity` |
| Halstead-Maße (Volume, Difficulty, Effort) | `radon.metrics` |
| Lines of Code (LOC, LLOC, SLOC) | `radon.raw` |
| Nesting Depth | Eigener AST-Walker mit Python `ast` |
| Number of Functions | AST-Visitor zählt `FunctionDef` / `AsyncFunctionDef` |
| Token Length | `tiktoken` (z.B. `cl100k_base`) |

- Speichern: `data/interim/classical_metrics.parquet`

### Schritt 7 — LM-CC implementieren (Hauptaufwand!)

Nach Xie et al. (2026), arXiv:2602.07882. Parameter aus dem Paper:

- **Modell:** CodeLlama-7B-HF (`meta-llama/CodeLlama-7b-hf`)
  - Alternativen ohne Gating: `bigcode/starcoder2-7b` oder `Qwen/Qwen2.5-Coder-7B`
- **Entropie-Schwellenwert:** τ = 0.67 (67%-Perzentil der Token-Entropien)
- **Gewichtungsfaktor:** α = 0.8

**Pipeline (Algorithmus 1 im Paper):**

1. **Preprocessing:** Kommentare und Docstrings entfernen
2. **Token-Entropie berechnen** mit lokalem LLM:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch

   tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/CodeLlama-7b-hf",
       torch_dtype=torch.float16,
       device_map="auto"  # nutzt MPS/CUDA automatisch
   )
   # Forward-Pass -> softmax über Logits -> Entropie pro Position
   # H(t_i) = -sum_j p(t_j | t<i) * log p(t_j | t<i)
   ```
3. **Entropien cachen** in `data/entropies/<instance_id>.npy` (einmaliger Aufwand)
4. **Semantische Einheiten** identifizieren:
   - Grenze, wenn Entropie > τ **oder** syntaktischer Delimiter (Loop-Ende, if-Ende, Funktions-Ende)
5. **Hierarchie aufbauen** (BFS): semantische Einheiten nach Einrückung partitionieren, rekursiv Unter-Einheiten erzeugen
6. **Features extrahieren** pro Hierarchie:
   - TotalCompLevel (Summe der Kompositionsebenen)
   - TotalBranch (Summe der Branching-Faktoren)
7. **LM-CC berechnen:**
   ```
   LM-CC = alpha * TotalBranch + (1 - alpha) * TotalCompLevel
   ```

**Hardware-Hinweise:**
- Mac mit Apple Silicon (M1–M4, ≥16 GB RAM): MPS-Backend, ~30–90 Min für 500 Snippets
- NVIDIA GPU (≥12 GB VRAM): ~10–20 Min
- Nur CPU: mehrere Stunden, aber einmalig
- Fallback: Google Colab (kostenlose GPU)

- Speichern: `data/processed/dataset_final.parquet`

### Schritt 8 — Daten joinen

- Klassische Maße + LM-CC + Predictions über `instance_id` mergen
- Pro Modell eine Spalte `resolved_<modellname>` (bool)
- Endgültiger Datensatz für die Analyse

### Schritt 9 — Analyse

- **Deskriptive Statistik:** Mittelwerte, Mediane, Verteilungen pro Komplexitätsmaß (Histogramme, Boxplots)
- **Korrelationsanalyse:**
  - Pearson (linear) + Spearman (monoton) zwischen jedem Maß und `resolved`
  - **Partielle Korrelation** unter Kontrolle der Code-Länge (wie im Paper, Gleichung 2) — wichtig, weil viele klassische Maße mit Länge konfundiert sind
  - Signifikanztests, Konfidenzintervalle
- **Subgruppen-Analyse** (Paper-Methode, Appendix C):
  - Samples nach Maß sortieren → in ~10 gleich große Gruppen einteilen
  - Pro Gruppe: Median des Maßes + Mean der Erfolgsrate
  - Spearman-Korrelation über die Gruppenwerte
- **Vergleich klassische Maße vs. LM-CC:** Welche Maße korrelieren stärker mit LLM-Erfolg?
- **Pro Modell separat** auswerten — gibt es Unterschiede zwischen stark/schwach?

### Schritt 10 — Visualisierung

- Korrelations-Heatmap (alle Maße × alle Modelle)
- Scatter-Plots: Komplexität vs. Erfolgsrate (mit gebinnten Werten)
- Boxplots: Komplexitätsverteilung resolved vs. not-resolved
- Plots nach `results/` speichern

### Schritt 11 — Schreiben

Parallel zur Implementierung, nicht erst am Ende:

- Früh (Woche 2–3): Einleitung, Grundlagen-Kapitel
- Mitte: Methodik (basiert auf umgesetzten Entscheidungen)
- Spät: Durchführung, Ergebnisse, Diskussion, Fazit
- Abschluss: Überarbeitung, Formatierung, Korrekturlesen

---

## Zeitplan (Richtwerte bei ~20h/Woche)

| Phase | Aufwand | Wochen |
|---|---|---|
| Setup + Datenbeschaffung | 12–18h | 1 |
| Code-Extraktion | 25–40h | 1,5–2 |
| Klassische Komplexitätsmaße | 8–12h | 0,5 |
| LM-CC-Implementierung | 25–40h | 1,5–2 |
| Analyse + Visualisierung | 15–25h | 1 |
| Schreiben | 40–70h | 2–3 (parallel) |
| Puffer | 15–25h | 1 |
| **Gesamt** | **~140–230h** | **10–12 Wochen** |

---

## Wichtige Entscheidungen (im Bericht begründen!)

- **SWE-Bench-Variante:** Verified (500 Tasks, hohe Qualität)
- **Modellauswahl für Erfolgs-Labels:** 2–3 Modelle vom Leaderboard, abdeckend stark/mittel/schwach
- **"Betroffener Codebereich":** enthaltende Funktion(en) der geänderten Zeilen
- **Filter:** Single-File-Patches, max. 3 Funktionen
- **Modell für Token-Entropie (LM-CC):** CodeLlama-7B-HF (wie im Paper) — Abweichungen dokumentieren
- **Hyperparameter LM-CC:** τ = 0.67, α = 0.8 (Paper-Werte)
- **Korrelationsmethode:** Spearman + partielle Korrelation unter Kontrolle der Code-Länge

---

## Theoretischer Hintergrund: Was ist LM-CC?

**Token-Entropie:** Maß für die Unsicherheit des LLM bei der Vorhersage des nächsten Tokens. Hohe Entropie = das Modell "stockt" → strukturell relevante Stelle im Code.

**Semantische Einheit:** Zusammenhängender Token-Block, in dem das LLM "ruhig durchliest". Entropie-Spikes oder syntaktische Delimiter (Loop-/Funktions-Ende) markieren Grenzen.

**Semantische Komposition:** Verschachtelung der Einheiten (flach vs. tief).

**LM-CC** = gewichtete Summe aus:
- **TotalCompLevel** (wie tief verschachtelt sind die Einheiten?)
- **TotalBranch** (wie viele alternative Pfade gibt es?)

Kerneinsicht des Papers: Nach Kontrolle der Code-Länge korrelieren klassische Maße (CC, Halstead, MI, CoC) **nicht** signifikant mit LLM-Performance — LM-CC dagegen sehr stark (Spearman r bis -0.97).

---

## Risikoposten

1. **LM-CC-Implementierung** — Algorithmus 1 im Paper genau umsetzen, hierarchische BFS-Zerlegung ist der kniffligste Teil
2. **CodeLlama-Zugang** — gated model, HuggingFace-Token nötig, ggf. Alternative wählen
3. **Hardware für Inferenz** — falls lokal nicht ausreichend, Colab einplanen
4. **SWE-Bench-Edge-Cases** — Patches mit ungewöhnlicher Struktur, fehlende Funktionen
5. **GitHub-API-Rate-Limits** — Token + Caching von Anfang an
6. **Schreiben dauert länger als geplant** — früh anfangen

---

## Status

- [ ] Schritt 1: Setup
- [ ] Schritt 2: SWE-Bench laden
- [ ] Schritt 3: Predictions holen
- [ ] Schritt 4: Patches parsen
- [ ] Schritt 5: Pre-Patch-Code extrahieren
- [ ] Schritt 6: Klassische Komplexitätsmaße
- [ ] Schritt 7: LM-CC implementieren
- [ ] Schritt 8: Daten joinen
- [ ] Schritt 9: Analyse
- [ ] Schritt 10: Visualisierung
- [ ] Schritt 11: Schreiben

---

## Kernreferenzen

- Xie, C., Shi, Y., Gu, X., Shen, B. (2026). *Rethinking Code Complexity Through the Lens of Large Language Models.* arXiv:2602.07882. — **LM-CC-Hauptpaper, Algorithmus, Parameter, Methodik**
- Jimenez, C. E., Yang, J., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
- McCabe, T. J. (1976). *A Complexity Measure.* IEEE TSE.
- Halstead, M. H. (1977). *Elements of Software Science.*
- Liu et al. (2024b). *Lost in the Middle: How Language Models Use Long Contexts.* — Theoretische Grundlage für LM-CC
- Cooper & Scholak (2024). *Perplexed: Understanding when LLMs are confused.* — Token-Entropie als Unsicherheitsmaß