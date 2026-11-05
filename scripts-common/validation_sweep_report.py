"""Reports a Remix settings sweep: validation messages per step and per-step performance.

A sweep (Developer tab > Settings Sweep > Run Settings Sweep) writes, in rtx-remix/logs:
  validation-sweep.log  the validation layer's messages in order with the
                        "[Sweep] step N/M: <name>: <overrides>" markers, each step
                        ending with a "[Sweep] messages:" count of every message ID
                        it saw (written only when a validation layer is loaded),
  sweep-perf.csv        one row per step: frames, fps, CPU and GPU frame time
                        percentiles, VRAM and shader compile wait,
  remix-dxvk.log        the markers, and with rtx.enableValidationLayers also the
                        runtime's own [VK_DEBUG_REPORT] lines.

For the messages it prints:
  - the message IDs seen in every step (always on),
  - the message IDs seen in only some steps, with those steps and counts,
  - one message text per ID.
For the performance it prints one table row per step, with the GPU p50
difference to the first (baseline) step.

Pure Python 3 stdlib.

Usage:
  python validation_sweep_report.py --capture validation-sweep.log
  python validation_sweep_report.py --perf sweep-perf.csv [--sort gpu]
  python validation_sweep_report.py --remix-log remix-dxvk.log
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict

STEP_RE = re.compile(r"\[Sweep\] step (\d+)/(\d+): (.*?): (.*)$")
DONE_RE = re.compile(r"\[Sweep\] done: (.*)$")
SUMMARY_HEAD_RE = re.compile(r"^\[Sweep\] messages: ")
SUMMARY_ID_RE = re.compile(r"^\[Sweep\]   (\d+) x (\S+) \((\w+)\)$")
VVL_RE = re.compile(r"^\[VVL\] (\w+) (\S+): (.*)$")
REMIX_LINE_RE = re.compile(r"^\[\d\d:\d\d:\d\d(?:\.\d+)?\] \w+:\s+")
REMIX_VK_RE = re.compile(r"\[VK_DEBUG_REPORT\] Code (-?\d+): (.*)$")
VUID_RE = re.compile(r"\[\s*([A-Za-z0-9_\-:.]+)\s*\]")
MSGID_RE = re.compile(r"MessageID = (0x[0-9a-fA-F]+)")
HANDLE_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
PRE_STEP = "0: before first step"
AFTER = "after sweep"


def read_lines(path):
  with open(path, "r", encoding="utf-8", errors="replace") as f:
    return f.read().splitlines()


def message_key(text):
  """Returns the VUID (or other bracketed ID) of a [VK_DEBUG_REPORT] message, or its message ID."""
  for m in VUID_RE.finditer(text):
    if "-" in m.group(1):
      return m.group(1)
  msgid = MSGID_RE.search(text)
  if msgid:
    return msgid.group(1).lower()
  # Messages with no ID are keyed by their text with handles removed.
  return HANDLE_RE.sub("0x?", text.strip())[:120]


def step_label(match):
  return "%s: %s" % (match.group(1), match.group(3))


class Sections:
  """Message counts per sweep step, in the order the steps ran."""

  def __init__(self):
    self.steps = OrderedDict()
    self.first_text = OrderedDict()
    self.severity = {}
    self.done = None
    self.current = PRE_STEP
    self._open(PRE_STEP, "")

  def _open(self, label, overrides):
    # A later pass of a looping sweep reuses the step names, and its counts add up.
    self.steps.setdefault(label, {"overrides": overrides, "ids": OrderedDict()})
    self.current = label

  def step(self, label, overrides):
    self._open(label, overrides)

  def finish(self, summary):
    self.done = summary
    self._open(AFTER, "")

  def add(self, key, count=1, severity=None, text=None):
    ids = self.steps[self.current]["ids"]
    ids[key] = ids.get(key, 0) + count
    if severity:
      self.severity.setdefault(key, severity)
    if text is not None:
      self.first_text.setdefault(key, text)
    elif key not in self.first_text:
      self.first_text[key] = ""

  def counts(self):
    total = {}
    for data in self.steps.values():
      for k, n in data["ids"].items():
        total[k] = total.get(k, 0) + n
    return total


def parse_capture(path):
  """Reads validation-sweep.log. The per-step summaries hold the exact counts, the [VVL] lines the texts."""
  sections = Sections()
  in_summary = False
  texts = {}
  for line in read_lines(path):
    if not line or line.startswith("#"):
      continue
    if SUMMARY_HEAD_RE.match(line):
      in_summary = True
      continue
    if in_summary:
      m = SUMMARY_ID_RE.match(line)
      if m:
        sections.add(m.group(2), int(m.group(1)), m.group(3), texts.get(m.group(2)))
        continue
      in_summary = False
    step = STEP_RE.search(line)
    if step:
      sections.step(step_label(step), step.group(4))
      continue
    done = DONE_RE.search(line)
    if done:
      sections.finish(done.group(1))
      continue
    m = VVL_RE.match(line)
    if m:
      texts.setdefault(m.group(2), m.group(3))
      sections.severity.setdefault(m.group(2), m.group(1))
  # An ID counted before any of its texts was read gets its text here.
  for k in sections.first_text:
    if not sections.first_text[k]:
      sections.first_text[k] = texts.get(k, "")
  return sections


def parse_remix_log(path):
  """Reads remix-dxvk.log: the step markers and the runtime's [VK_DEBUG_REPORT] messages."""
  sections = Sections()
  current = None

  def flush():
    if current is not None:
      text = "\n".join(current)
      sections.add(message_key(text), 1, None, text)

  for raw in read_lines(path):
    is_new_line = REMIX_LINE_RE.match(raw) is not None
    line = REMIX_LINE_RE.sub("", raw)
    if not is_new_line and current is not None:
      current.append(raw)
      continue
    flush()
    current = None
    step = STEP_RE.search(line)
    if step and "[VK_DEBUG_REPORT]" not in line:
      sections.step(step_label(step), step.group(4))
      continue
    done = DONE_RE.search(line)
    if done:
      sections.finish(done.group(1))
      continue
    vk = REMIX_VK_RE.search(line)
    if vk:
      current = [vk.group(2)]
  flush()
  return sections


def print_messages(name, sections, max_text):
  steps = sections.steps
  counts = sections.counts()
  sweep_steps = [s for s in steps if s not in (PRE_STEP, AFTER)]
  print("=" * 100)
  print("Source: %s: %d steps, %d message IDs, %d messages" % (name, len(sweep_steps), len(counts), sum(counts.values())))
  if sections.done:
    print("Sweep summary: %s" % sections.done)
  if not sweep_steps:
    print("No [Sweep] step markers in this file; every message is listed under '%s'." % PRE_STEP)

  always = [k for k in counts if sweep_steps and all(k in steps[s]["ids"] for s in sweep_steps)]
  some = [k for k in counts if k not in always]
  some.sort(key=lambda k: -counts[k])

  def sev(k):
    return (" " + sections.severity[k]) if k in sections.severity else ""

  print("\n-- In every step (%d) --" % len(always))
  for k in always:
    print("  %s%s  (%d total)" % (k, sev(k), counts[k]))

  print("\n-- In some steps (%d) --" % len(some))
  for k in some:
    print("  %s%s  (%d total)" % (k, sev(k), counts[k]))
    for s in steps:
      if k in steps[s]["ids"]:
        print("      %s  x%d" % (s, steps[s]["ids"][k]))

  print("\n-- Per step --")
  for s, data in steps.items():
    if not data["ids"] and s in (PRE_STEP, AFTER):
      continue
    new = [k for k in data["ids"] if k not in always]
    print("  %s  [%s]" % (s, data["overrides"] or "-"))
    print("      %d IDs, %d messages, %d IDs not in every step%s" %
          (len(data["ids"]), sum(data["ids"].values()), len(new), (": " + ", ".join(new)) if new else ""))

  print("\n-- One message per ID --")
  for k in counts:
    text = sections.first_text.get(k) or "(no text written)"
    if len(text) > max_text:
      text = text[:max_text] + " ..."
    print("\n[%s]" % k)
    print("  " + text.replace("\n", "\n  "))
  print()


def to_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def print_perf(path, sort):
  notes = []
  with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
    lines = []
    for line in f:
      if line.startswith("#"):
        notes.append(line[1:].strip())
      else:
        lines.append(line)
  rows = list(csv.DictReader(lines))
  print("=" * 100)
  print("Performance: %s, %d steps" % (path, len(rows)))
  for note in notes:
    print(note)
  if any(r.get("validation_layer") == "1" for r in rows):
    print("WARNING: a validation layer was active, so these figures are not representative.")
  if not rows:
    return

  baseline = to_float(rows[0].get("gpu_frame_ms_p50"))
  if sort == "gpu":
    rows.sort(key=lambda r: -(to_float(r.get("gpu_frame_ms_p50")) or 0.0))
  elif sort == "fps":
    rows.sort(key=lambda r: to_float(r.get("avg_fps")) or 0.0)

  def cell(value, fmt):
    v = to_float(value)
    return fmt % v if v is not None else "-"

  header = ("step", "pass", "fps", "cpu p50", "cpu p95", "gpu p50", "gpu p95", "gpu vs base", "VRAM MiB", "compile s", "name")
  print("\n%5s %4s %7s %8s %8s %8s %8s %11s %9s %9s  %s" % header)
  for r in rows:
    gpu = to_float(r.get("gpu_frame_ms_p50"))
    delta = "%+.2f" % (gpu - baseline) if gpu is not None and baseline is not None else "-"
    print("%5s %4s %7s %8s %8s %8s %8s %11s %9s %9s  %s" % (
      r.get("step", ""), r.get("pass", ""),
      cell(r.get("avg_fps"), "%.1f"),
      cell(r.get("cpu_frame_ms_p50"), "%.2f"), cell(r.get("cpu_frame_ms_p95"), "%.2f"),
      cell(r.get("gpu_frame_ms_p50"), "%.2f"), cell(r.get("gpu_frame_ms_p95"), "%.2f"),
      delta, r.get("vram_mib", ""), cell(r.get("compile_wait_s"), "%.1f"), r.get("name", "")))
  print()


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--capture", help="validation-sweep.log written by the sweep")
  parser.add_argument("--remix-log", help="remix-dxvk.log of the same run")
  parser.add_argument("--perf", help="sweep-perf.csv written by the sweep")
  parser.add_argument("--sort", choices=("step", "gpu", "fps"), default="step", help="order of the performance table")
  parser.add_argument("--max-text", type=int, default=600, help="characters of message text to print per ID")
  args = parser.parse_args()

  if not (args.capture or args.remix_log or args.perf):
    parser.error("give --capture, --remix-log and/or --perf")

  if args.capture:
    print_messages("validation-sweep.log", parse_capture(args.capture), args.max_text)
  if args.remix_log:
    print_messages("remix-dxvk.log", parse_remix_log(args.remix_log), args.max_text)
  if args.perf:
    print_perf(args.perf, args.sort)
  return 0


if __name__ == "__main__":
  sys.exit(main())
