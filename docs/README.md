# LCM Documentation Guide

This directory contains design documents and specifications in both Chinese and English.

---

## Document Index

| File | Content | Audience |
|------|---------|----------|
| `a.md` / `a_cn.md` | **Architecture Design Document** — Project overview, technical background, system architecture, detailed module design, training loss, hardware efficiency, roadmap | All readers, starting point |
| `b.md` / `b_cn.md` | **Technical Specification & Implementation Guide** — Notation conventions, encoder design, mathematical definitions of all six lattices, training loss and parameter updates, system integration pseudocode | Developers, researchers |
| `c.md` / `c_cn.md` | **Zero-Parameter Inference Engine Specification** — Dynamic dataflow graph architecture, operation primitives, DAG construction and execution flow, external interface definitions | Inference engine developers |
| `d.md` / `d_cn.md` | **Safety Subsystem Specification** — Danger lattice detection mechanism, external verifier, self-lattice activation preconditions, conflict abort protocol | Security auditors |
| `e.md` / `e_cn.md` | **Self-Awareness Research** — Self-observation stack, black box recorder, narrative memory, reflection loop, causal subject, self lattice, behavior exploration, hierarchical model | Cognitive science researchers |
| `training_report_v1_en.md` / `training_report_v1_cn.md` | **V1 Training Experiment Report** — Full experimental records from GTX 1650 to RTX 5090, training curves, key findings | Developers, researchers |

## Reading Guide

- **New to LCM**: Start with `a.md` for the overall architecture and design philosophy.
- **Implementing or modifying lattices**: `b.md` contains complete mathematical definitions and implementation details for all lattices.
- **Inference performance**: `c.md` describes the zero-parameter inference engine design and interfaces.
- **Safety audit**: `d.md` defines the three-layer safety architecture and hard-abort protocol.
- **Self-awareness & metacognition**: `e.md` covers self-observation, reflection loops, causal subjects, and other high-level modules.

---

---

> **中文版**: [README_cn.md](README_cn.md) — 中文文档导航。

## Version

This documentation corresponds to the LCM v3.0 architecture specification. Documents are updated in sync with the source code.
