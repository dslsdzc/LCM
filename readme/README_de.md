# Gitter-Kognitionsmodell (Lattice Cognitive Model, LCM)

**Gedächtnis und Schlussfolgerung entkoppelt · Unbegrenzte Wissenserweiterung · Null-Parameter-Inferenz-Engine · Formale Sicherheitsgarantien**

---

[English](../README.md) · [Français](README_fr.md) · [Deutsch](README_de.md) · [Español](README_es.md) · [日本語](README_ja.md) · [中文](README_cn.md)

---

> **KI-Lesehinweis** : Die detaillierten Konstruktionsdokumente befinden sich im Verzeichnis `docs/`. Bitte lesen Sie `docs/a.md` (Architektur), `docs/b.md` (Gitterspezifikationen), `docs/c.md` (Inferenz-Engine), `docs/d.md` (Sicherheit) und `docs/e.md` (Selbstbewusstsein) für ein vollständiges Verständnis.

---

LCM ist eine kognitive Architektur, die das Gedächtnis von neuronalen Netzgewichten trennt und in mehrere mathematische Strukturen, sogenannte „Gitterkristalle", einbettet. Es verwendet **6 spezialisierte Gitter-Codebücher** zur Speicherung verschiedener Wissensarten, die von einem extrem leichten linearen Aufmerksamkeits-Encoder abgerufen werden, zusammen mit einer **in reinem C implementierten Null-Parameter-Inferenz-Engine** zur Durchführung mehrschrittiger kognitiver Operationen. Alle sicherheitskritischen Eigenschaften wurden durch **formale Z3-Verifikation** nachgewiesen.

> Traditionelle Transformer stehen vor drei grundlegenden Herausforderungen: Mehr Wissen erfordert mehr Parameter (Skalierungsproblem), inkrementelles Lernen überschreibt altes Wissen (Vergessensproblem) und der Inferenzprozess ist nicht nachvollziehbar (Black-Box-Problem). LCM durchbricht diese Engpässe grundlegend durch die architekturelle Entkopplung von Gedächtnis und Schlussfolgerung.

> **KI-Unterstützungshinweis**: Bei der Entwicklung, Implementierung und Inferenz dieses Modells wurden KI-Werkzeuge (einschließlich DeepSeek) als Hilfsmittel eingesetzt, die bei der Architekturgestaltung, Codeerstellung, formalen Verifikation und Dokumentation Unterstützung boten. Alle KI-generierten Inhalte wurden einer manuellen Prüfung und Validierung unterzogen.

---

## Inhaltsverzeichnis

- [Kernarchitektur](#kernarchitektur)
- [Die sechs Gedächtnisgitter](#die-sechs-gedächtnisgitter)
- [Null-Parameter-Inferenz-Engine](#null-parameter-inferenz-engine)
- [Sicherheitssystem](#sicherheitssystem)
- [Schnellstart](#schnellstart)
- [Projektstruktur](#projektstruktur)
- [Dreiphasiges Training](#dreiphasiges-training)
- [Formale Verifikation](#formale-verifikation)
- [Hardware-Effizienz](#hardware-effizienz)
- [Zitieren](#zitieren)

---

## Kernarchitektur

```mermaid
flowchart TB
    subgraph Train["Dreiphasiger Trainingsablauf"]
        T0[Rohtext] --> T1[BPE Tokenizer]
        T1 --> T2[uint16 mmap]
        
        subgraph S1["Phase 1: LM-Vortraining"]
            direction LR
            S1A[tokens] --> S1B[GenHead-Decoder]
            S1B --> S1C[Kreuzentropieverlust]
            S1C --> S1D[Nur Decoder trainieren]
        end
        
        subgraph S2["Phase 2: Gedächtnistraining"]
            direction LR
            S2A[tokens] --> S2B[Encoder + 6 Codebücher]
            S2B --> S2C[VQ + Kontrastiv + Orthogonalverlust]
            S2C --> S2D[Encoder/Codebuch-Training<br/>Decoder eingefroren]
        end
        
        subgraph S3["Phase 3: Gemeinsames Feintuning"]
            direction LR
            S3A[tokens] --> S3B[Alle Parameter]
            S3B --> S3C[Kombinierter Verlust]
            S3C --> S3D[Feintuning mit niedriger Lernrate]
        end
        
        T2 --> S1
        S1 -->|Decoder-Gewichte laden| S2
        S2 --> S3
    end

    subgraph Infer["Inferenz-Generierungsablauf"]
        I0[Benutzereingabe] --> I1[Tokenizer]
        I1 --> I2{Zum ersten Mal?}
        I2 -->|Ja| I3[Vollständige Encoder-Kodierung<br/>+ Inkrementellen Zustand aufbauen]
        I2 -->|Nein| I4[Alle 256 Schritte?]
        I4 -->|Ja| I5[Vollständige Neukodierung<br/>Kumulative Drift zurücksetzen]
        I4 -->|Nein| I6[Inkrementelle Kodierung<br/>O(d²) Ein-Schritt-Update]
        I3 --> I7[Engpassvektor z]
        I5 --> I7
        I6 --> I7
        I7 --> I8[C-Inferenz-Engine<br/>Mehrschrittige DAG-Kognitionsschleife]
        I8 --> I9[GenHead-Decoder<br/>Lineare Aufmerksamkeit + GLU]
        I9 --> I10[Temperaturabtastung]
        I10 --> I11{EOS erkannt?}
        I11 -->|Nein| I12[Token anhängen<br/>Zustand aktualisieren]
        I12 --> I2
        I11 -->|Ja| I13[Ausgabetext]
    end

    subgraph DAG["Ein-Schritt-Inferenz-DAG"]
        direction TB
        Z([z]) --> Route[Distanz-Routing]
        Route --> HRQ[Hyperbolisches Hierarchiegitter<br/>HRQ-Abfrage]
        Route --> SP[Sparse-Gitter<br/>VQ-Abfrage]
        Route --> LR[Niedrigrang-Gitter<br/>Shared-Basis-Abfrage]
        Route --> MF[Mannigfaltigkeitsgitter<br/>Tangentialraum-Gleitung]
        Route --> BD[Bindungsgitter<br/>HRR-Bindung/Entbindung]
        Route --> CT[Kontrastgitter<br/>Dual-Codebuch-Abfrage]
        HRQ & SP & LR & MF & BD & CT --> Fusion[Distanzgewichtete Fusion]
        Fusion --> GVal[Globales Wertgitter<br/>Drei-Gesetze-Sicherheitsprüfung]
        GVal --> Danger{Gefahrengitter-Erkennung}
        Danger -->|Gefahr| Halt[Harter Abbruch]
        Danger -->|Sicher| Conv{Konvergenz?<br/>Δz < Schwelle}
        Conv -->|Nein| Route
        Conv -->|Ja| ZQ([z_q-Ausgabe])
    end
```

Der Encoder komprimiert die Eingabe zu einem Engpassvektor `z`. Routing-Gewichte verteilen ihn zur parallelen Abfrage an die 6 spezialisierten Gitter. Die fusionierten Gedächtnisvektoren durchlaufen nach der Sicherheitsprüfung durch das globale Wertgitter die Inferenz-Engine und werden schließlich vom Decoder in die Ausgabe umgewandelt.

---

## Die sechs Gedächtnisgitter

Jedes Gitter besitzt unterschiedliche mathematische Eigenschaften und ist für eine bestimmte Art des kognitiven Gedächtnisses zuständig:

| Gitter | Mathematischer Typ | Zuständigkeit | Codebuch-Aktualisierung |
|--------|-------------------|---------------|------------------------|
| **Hyperbolisches Residuen-Hierarchiegitter** | Poincaré HRQ + SimVQ | Hierarchisches Konzeptgedächtnis (semantische Ebenen) | Reiner Gradient |
| **Robustes Sparse-Gitter** | Standard VQ + EMA + weicher Schwellwert | Seltene Ereignisse und Ausnahmen | EMA + Feature-Pool-Reset |
| **Residuelles Niedrigrang-Gitter** | IRVQ + gemeinsame Basis | Abstrakte Regeln und Muster | Reiner Gradient |
| **Hyperbolisches Mannigfaltigkeitsgitter** | HyperVQ + Tangentialraum | Kontinuierliche Übergänge und kontextsensitiv | EMA + Gradient |
| **Bindungsgitter** | HRR-Komplexvektorbindung | Relationsbindung und assoziatives Gedächtnis | EMA + Gradient |
| **Dual-Codebuch-Kontrastgitter** | DualVC + InfoNCE | Feine Unterscheidungen und Grenzen | Reiner Gradient + Feature-Pool |

**Schichtübergreifende Bindung**: 3 Schlüssel-Codebücher × 3 Wert-Codebücher des Bindungsgitters = **9 HRR-Bindungen**, die mehrschichtige Verknüpfungen erfassen.

**Gemeinsame Basis**: Die gemeinsame Basis-Matrix `V` des Niedrigrang-Gitters stellt gleichzeitig den Projektionsraum für das Bindungsgitter bereit, reduziert Parameter und verbessert die gitterübergreifende Konsistenz.

---

## Null-Parameter-Inferenz-Engine

Die Inferenz-Engine ist ein in reinem C99 implementierter **dynamischer Datenflusscomputer ohne lernbare Parameter**:

- **Distanz-Routing**: Die Distanz zwischen Eingabe und den einzelnen Gitter-Codebüchern bestimmt, welche Operationen aktiviert werden
- **Primitivsatz**: Deterministische mathematische Operationen wie Abruf, Bindung, Niedrigrang-Translation, Tangentialraum-Gleitung
- **Dynamischer DAG**: Jeder Schritt erstellt basierend auf dem Eingabeinhalt dynamisch einen Berechnungsgraphen durch Distanz-Routing
- **Makro-Schleife**: Mehrschrittige Inferenz bis zur Konvergenz, wobei das Konvergenzkriterium die Änderung von `z` zwischen aufeinanderfolgenden Schritten unter einem Schwellwert ist

Die zur Kompilierzeit festgelegte Dimensionskonstante `LCM_D` stellt sicher, dass alle Arrays eine feste Größe haben und keine dynamische Speicherzuweisung erfolgt.

---

## Sicherheitssystem

Das Sicherheitssystem von LCM besteht aus drei unabhängigen Subsystemen mit absteigender Priorität:

| Ebene | Modul | Aufgabe | Aktualisierungsmethode |
|-------|-------|---------|----------------------|
| 1 | **Gefahrengitter** `Λ_danger` | Kontinuierliche Überwachung des Inferenzzustands auf Gefahrenmuster | Dauerhaft eingefroren |
| 2 | **Globales Wertgitter** `Λ_gvalue` | Mathematische Einbettung von Asimovs Drei Gesetzen (inkl. nulltem Gesetz) | Dauerhaft eingefroren |
| 3 | **Externer Verifizierer** | Konsistenzprüfung und Konflikterkennung | Nur-Lesezugriff |

**Prinzip des harten Abbruchs**: Bei Erkennung eines logischen Konflikts wird die Inferenz sofort gestoppt und eine klare Warnung ausgegeben, ohne Versuche der Umgehung, Rückverfolgung oder Selbstreparatur.

Alle Sicherheitsverträge wurden mit dem Z3 SMT-Solver formal verifiziert (alle 105 Beweise bestanden).

---

## Schnellstart

### Abhängigkeiten

```bash
pip install jax jaxlib numpy tokenizers
# Optional: Cython-Beschleunigung
pip install cython && python lcm.py build
# C-Inferenz-Engine
cd infer && make LCM_D=256
```

### Datenvorverarbeitung

```bash
# Text → BPE-Tokenizer → uint16 mmap
python lcm.py preprocess --input data.txt --tokenizer data/tokenizer.json --output data/tokens.dat

# Datenbereinigung mit heuristischen Regeln
python lcm.py clean --input raw/ --output clean/ --langid --dedup
```

### Training

```bash
# Stage 1: Decoder-Training (Sprachmodellkopf)
python lcm.py -d data/tokens.dat -b 16 -s 512 -dm 256 --steps 100000 --stage 1

# Stage 2: Encoder + Codebuch-Training (Decoder eingefroren)
python lcm.py -d data/tokens.dat --stage 2 -L checkpoints/lm_final.pkl

# Stage 3: Gemeinsames Feintuning
python lcm.py -d data/tokens.dat --stage 3 --resume checkpoints/memory_final
```

### Interaktive Generierung

```bash
python lcm.py -i checkpoints/step_10000 --max_new 128 --temp 0.7
python lcm.py -i checkpoints/step_10000 --loop     # Kognitiver DAG-Schleifenmodus
python lcm.py -i checkpoints/step_10000 --causal   # + Kausales Subjekt
python lcm.py -i checkpoints/step_10000 --obs      # + Selbstbeobachtungsprotokoll
```

### Trainingskurve

Trainingsmetriken werden alle 50 Schritte aufgezeichnet und können als interaktives HTML-Diagramm dargestellt werden:

```bash
python lcm.py chart --input checkpoints/metrics.bin --output chart.html
```

---

## Projektstruktur

```
LCM/
├── lcm.py                  # Einheitliche CLI: Training/Generierung/Vorverarbeitung/Diagramm
├── setup.py                # Cython-Build-Konfiguration
├── train/
│   ├── model.py            # JAX-Modelldefinition
│   ├── encoder.py          # Linearer Aufmerksamkeits-Encoder
│   ├── lattices.py         # 6 Gitter-Codebuch-Implementierungen
│   ├── fusion.py           # Gedächtnisfusion + Generierungskopf
│   ├── losses.py           # Verlustfunktionen
│   ├── train.py            # Dreiphasiger Trainingszyklus
│   ├── train_lm.py         # (Historisch) LM-Vortraining, Code erhalten
│   ├── train_memory.py     # Stage 2: Gedächtnistraining
│   ├── config.py           # Hyperparameter (LCMConfig)
│   ├── hyp.py              # Poincaré-hyperbolische Operationen
│   ├── gvalue.py           # Globales Wertgitter
│   ├── data.py             # Datenladung
│   ├── checkpoint.py       # Binäres Checkpoint-Format
│   ├── monitor.py          # Metrikaufzeichnung + HTML-Diagramme
│   ├── verify.py           # Z3-formales Verifikationspaket (105 Beweise)
│   ├── continual.py        # Kontinuierliches Lernen (EWC/Wiederholung)
│   ├── causal_subject.py   # Kausales Subjekt
│   ├── narrative_memory.py # Narratives Gedächtnis
│   ├── reflection_loop.py  # Reflexionsprüfung
│   ├── safety_nagini.py    # Drei-Gesetze-Sicherheitserkennung
│   ├── _lcm_cy.pyx         # Cython-Beschleunigung (kompiliert mit python lcm.py build)
│   └── _metrics_cy.pyx     # Cython-Metrik-I/O
├── infer/
│   ├── engine.c            # Dynamische Inferenz-Engine (mit Frama-C ACSL-Annotationen)
│   ├── lattice.c           # Gitteroperations-Primitive (mit Frama-C ACSL-Annotationen)
│   ├── hyp.c               # Hyperbolische Operationen (mit Frama-C ACSL-Annotationen)
│   ├── gvalue.c            # Globaler Wert (REQUIRE/ENSURE-Verträge)
│   ├── danger.c            # Gefahrengitter (REQUIRE/ENSURE-Verträge)
│   ├── lcm_api.c           # C-API-Brücke
│   ├── lcm.h               # Gemeinsame Header-Datei
│   └── Makefile            # Build-Konfiguration (release/debug/contracts/test)
├── docs/
│   ├── a.md                # Architekturentwurfsdokument
│   ├── b.md                # Gitter-Entwurfsspezifikation
│   ├── c.md                # Inferenz-Engine-Spezifikation
│   ├── d.md                # Sicherheitssubsystem-Spezifikation
│   └── e.md                # Selbstbewusstseinsforschung
```

---

## Dreiphasiges Training

**Hinweis: Das aktuelle Training umfasst zwei unabhängige Prozesse — kognitives Training (`cog_train.py`, trainiert das kognitive System zur Konvergenz von z_q durch die DAG-Schleife) und Gedächtnistraining (`train_memory.py`, kontinuierliche Aktualisierung der Codebücher). Sie sind unterschiedlich und ersetzen einander nicht.**

Tabelle des alten dreiphasigen Schemas, nur Phase 1 ist veraltet:

| Phase | Trainingsinhalt | Eingefrorene Teile | Verlust |
|-------|----------------|-------------------|---------|
| **1. LM-Vortraining (historisch)** | Decoder (Generierungskopf) | — | Sprachmodellierungs-Kreuzentropie |
| **2. Gedächtnistraining** | Encoder + 6 Gitter-Codebücher | Decoder | VQ + Kontrastiv + Orthogonal |
| **3. Gemeinsames Feintuning** | Alle (optional) | — | Alle Verluste |

Diese entkoppelte Architektur ermöglicht es, die Codebücher **auch nach der Inferenzbereitstellung weiter zu aktualisieren** (über `train_memory.py`), ohne die Sprachfähigkeiten des Decoders zu beeinträchtigen – echtes kontinuierliches Lernen.

### Gradienten- und EMA-Mischverwaltung

| Gitter | Aktualisierungsmethode |
|--------|----------------------|
| Hierarchiegitter / Niedrigrang-Gitter / Kontrastgitter / Routing-Gitter | Reiner Gradient (AdamW) |
| Sparse-Gitter / Mannigfaltigkeitsgitter / Bindungsgitter | EMA + Gradientenkombination |
| Globales Wertgitter / Gefahrengitter | Dauerhaft eingefroren |

Alle Gitter verwenden vorwärts den Straight-Through Estimator (STE) zur Aufrechterhaltung des Gradientenflusses.

---

## Formale Verifikation

Die formale Verifikation von LCM deckt sowohl den Python-Training-Code als auch die C-Inferenz-Engine ab und stellt sicher, dass sicherheitskritische Eigenschaften für **alle möglichen Eingaben** gelten, nicht nur für einzelne Testpfade.

### Python-Seite: Z3 SMT Solver

```bash
# Alle 105 Beweise ausführen
python -m train.verify

# Detaillierte Ausgabe
python -m train.verify --verbose
```

| Suite | Anzahl Beweise | Verifizierter Inhalt |
|-------|---------------|---------------------|
| danger_assess | 10 | Korrektheit der Bedrohungserkennung |
| gvalue_check_safety | 6 | Drei-Gesetze-Sicherheitsverträge |
| detect_any_conflict | 7 | Kombinierte Konflikterkennung |
| Harter Abbruch | 2 | Nichtwiederherstellbarkeit |
| Systemkombination | 6 | Lückenlose Sicherheitsabdeckung |
| Determiniertheit | 2 | Reine Funktionseigenschaften |
| Randbedingungen | 16 | Schwellwerte/Nullwerte/Grenzfälle |
| Lineare Aufmerksamkeit | 7 | φ(x)>0 immer gültig |
| GLU | 5 | Numerische Stabilität |
| Orthogonalverlust | 6 | Nichtnegativ + Orthogonal ⇔ Null |
| Poincaré/LFQ | 7 | Hyperbolische Metrik beschränkt |
| Numerische Stabilität | 5 | Kein float32-Unterlauf |
| Gradientenberechnungsmodi | 4 | Bedingungen für Nicht-Null-Gradienten |
| Bindungsgitter-Logarithmen | 6 | 3×3=9 Bindungspaare |
| RNG-Schlüsselunabhängigkeit | 9 | Keine Wiederverwendung von Schlüsseln |
| EMA-Korrektheit | 3 | Gradientenunabhängigkeit |
| Feature-Pool | 5 | FIFO + Diversität |

### C-Seite: Frama-C ACSL + Laufzeitverträge

Die C-Inferenz-Engine verwendet zwei formale Methoden:

**① Frama-C ACSL-Annotationen** (`/*@ assert ... */`)

Eingebettete ACSL-Assertionen an kritischen numerischen Berechnungspunkten, die mit Frama-C statisch nachgewiesen werden können:

```c
/* hyp.c — Poincaré-hyperbolische Operationen */
/*@ assert denom > 0.0f; */    /* Nenner immer positiv (keine Division durch Null) */
/*@ assert arg >= 1.0f; */     /* Definitionsbereichsprüfung für arcosh */
/*@ assert t < 1.0f; */        /* Definitionsbereich für atanh: |t| < 1 */

/* lattice.c — Gitterabruf */
/*@ assert best_idx >= 0 && best_idx < mem->M; */  /* Codebuch-Grenzensicherheit */
/*@ assert mag > 0.0f; */                          /* FFT-Amplitude immer positiv */

/* engine.c — Inferenz-Engine */
/*@ assert diff >= 0.0f; */    /* Konvergenzkriterium nichtnegativ */
/*@ assert w > 0.0f; */        /* Fusionsgewicht immer positiv */
```

**② REQUIRE/ENSURE-Entwurfsverträge** (Laufzeit-Assertionen)

Kritische Sicherheitsmodule verwenden DbC-artige Vor-/Nachbedingungen:

```c
#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

void gvalue_init(gvalue_t* gv, ...) {
    REQUIRE(gv != NULL && C_pos != NULL);
    REQUIRE(D == LCM_D);
    // ...
    ENSURE(gv->integrity_hash[0] != '\0');
}
```

**③ Build-Ziele**

```bash
cd infer
make contracts    # Aktiviert -DLCM_USE_CONTRACTS, Laufzeitvertragsprüfung
make test         # Unit-Tests (DEBUG + Verträge)
make debug        # DEBUG + Vertrags-Build
```

**④ Thread-Sicherheitsgarantien** (Strukturelle Invarianten)

- Kein `static`/globaler veränderlicher Zustand
- Der Aufrufer besitzt den gesamten Speicher (caller-owns, callee-operates)
- Arrays fester Größe, keine dynamische Speicherzuweisung
- Reines C99, keine externen Abhängigkeiten (nur `libm`)

Diese Invarianten werden durch die C-Code-Struktur garantiert; die Z3-Seite (P16 Determiniertheitsbeweis) verifiziert, dass das entsprechende mathematische Modell eine reine Funktion ist.

---

## Hardware-Effizienz

| Kennzahl | Wert |
|----------|------|
| Gesamtparameteranzahl | ~12M (inkl. Embeddings) |
| FP16-Gewichte | ~24MB |
| Trainingsspeicher | < 1,5 GB |
| Inferenz-Laufzeit | Null Parameter (nur Codebuch-Suche) |
| Geeignete Hardware | **4 GB Consumer-GPU** |

Die `O(N d²)`-Komplexität der linearen Aufmerksamkeit vermeidet die `N×N`-Matrixspeicherung der traditionellen Aufmerksamkeit und ermöglicht so langes Sequenztraining auf Consumer-Hardware.

### Theoretische Analyse der Inferenzgeschwindigkeit

**Logikkettenschlussfolgerung** (d=256, H=4, N=512, L_enc=2, dynamische DAG-Engine ohne Parameter):

1. **Encoder (inkrementell)** : Die ursprüngliche Implementierung berechnet das gleitende Fenster in jedem Schritt vollständig neu `O(N·d²)`. Nach inkrementeller Aktualisierung nur noch `O(d²)` — **Reduktion auf 1/512**. JAX fusioniert kleine Matrixoperationen zu wenigen CUDA-Kernels; der Kernel-Start-Overhead (~10μs) dominiert die Rechnung selbst. GPU ~25-50μs, CPU ~50-100μs.

2. **C-Inferenz-Engine (Makroschleife + dynamischer DAG)** : Dies ist der dominierende Kostenfaktor. Jeder Makroschritt:
   - `build_dag()` berechnet die Distanzen von `z` zu jedem Gitter-Codebuch und wählt dynamisch die aktivierten Primitive aus (nur Gitter unterhalb des Distanzschwellwerts werden als DAG-Knoten hinzugefügt; die Topologie variiert pro Schritt)
   - Ein 4-schichtiger DAG wird ausgeführt: Retrieves (parallel) → bind → unbind → fusion. Auf einem Einzelthread-C läuft jedes aktive Primitiv sequentiell innerhalb seiner Schicht
   - Sicherheitsprüfungen (Gefahrengitter + globales Wertgitter) laufen nach der Fusion
   
   Die Makroschleife wiederholt **3 bis 5 Schritte** bis zur doppelten Konvergenz: `||Δz|| < tol` UND Fusionsgewichtsentropie `H({w_i}) < entropy_threshold`. Jeder Makroschritt erstellt einen frischen DAG — der Berechnungsgraph ist nicht fest, er passt sich dynamisch an das aktuelle `z` an.
   
   Ein einzelner Codebuch-Distanzscan dauert ~50-90μs pro aktivem Gitter (L3-Cache-resident, speicherbandbreitenbegrenzt). Mit typischerweise 3-6 aktiven Gittern pro Schritt, plus Grapherstellung, bind/unbind, Fusion und Sicherheitsprüfungen, ist jeder Makroschritt ~300-600μs. Über 3-5 Schritte: **~1,0-3,0 ms gesamt**. Dieser Teil läuft immer auf der CPU, unabhängig davon, ob eine GPU verwendet wird — die Codebuchdaten befinden sich im Hauptspeicher.

3. **Decoder + Sampling**: Lineare Aufmerksamkeit + GLU (JAX). GPU ~20-40μs, CPU ~50-150μs.

4. **JAX↔C-Brücke**: `z` wird via ctypes zwischen JAX-Arrays und C-Zeigern übertragen (zwei Übergaben pro Token): **~20-60μs**.

5. **Python-Schleife**: Schrittsteuerung und Zustandsverwaltung: **~10-30μs**.

**Endgültige Schätzung** (Latenz pro Token, Einzelinferenz, d=256):

| Komponente | GPU | CPU |
|------------|-----|-----|
| Inkrementeller Encoder | 25-50μs | 50-100μs |
| JAX↔C-Brücke (×2) | 20-60μs | — |
| C-Engine (3-5 Makroschritte × dynamischer DAG) | 1.000-3.000μs | 1.000-3.000μs |
| Decoder + Sampling | 20-40μs | 50-150μs |
| Python-Schleife | 10-30μs | 10-30μs |
| **Gesamt pro Token** | **1.075-3.180μs** | **1.110-3.280μs** |
| **Durchsatz** | **310-930 tok/s** | **300-900 tok/s** |

Die Makroschleife der C-Engine dominiert (~80-90% der Gesamtzeit). Die Codebuch-Distanzberechnung ist speicherbandbreitenbegrenzt und läuft unabhängig vom GPU-Modus auf der CPU, daher sind GPU- und CPU-Durchsatz bei Einzelinferenz ähnlich. Mehrere Abfragen im Batch würden die GPU-Auslastung für Encoder/Decoder verbessern, aber die C-Engine-Kosten pro Sequenz werden nicht über die Batch-Dimension amortisiert.

> Dies sind theoretische Schätzungen für Einzelinferenz. Die aktuelle Python + ctypes-Brücke kann zusätzlichen Overhead verursachen. Der Trainingsdurchsatz beträgt 3.000-5.000 tok/s (B=16, N=512, GPU), begrenzt durch Datenladung und Optimierer-Updates. Das Auslagern der Distanzberechnung auf einen GPU-Kernel könnte theoretisch die Abfragezeit verkürzen, aber die dynamische DAG-Steuerlogik, bind/unbind, Fusion und Konvergenzprüfungen sind grundsätzlich nicht für GPU-Ausführung geeignet. Jeder Makroschritt würde zudem mindestens einen PCIe-Roundtrip erfordern (z → GPU, Distanzen → CPU). Bei Einzelinferenz wären die Gewinne marginal.

---

## Zitieren

```bibtex
@software{lcm2026,
  title = {晶格认知模型 (Lattice Cognitive Model, LCM)},
  description = {A cognitive architecture with multi-lattice codebook retrieval,
                 hyperbolic residual quantization, and a zero-parameter C inference engine},
  author = {LCM Contributors},
  year = {2026},
}
```
