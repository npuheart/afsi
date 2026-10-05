#!/usr/bin/env python
"""Generate a new IB2d demo from the merged template.

    python port_example.py <NNN> <upstream_relative_dir>

``<NNN>`` is the demo number (``demo_480``) and ``<upstream_relative_dir>`` is
the example directory relative to ``third_party/ib2d/matIB2d/Examples`` (or an
absolute path).  The script

1. refuses to run when the target demo directory already exists;
2. refuses to run when the upstream directory has no ``input2d``;
3. creates ``demo_NNN/ib2d_input/`` with every file the example needs, copied
   verbatim (``input2d``, all ``<string_name>.*`` data files, the example's own
   ``*.m``/``*.C`` user modules and any small auxiliary text data file);
4. copies the template code (``main.py``, ``compare.py``, ``ib2d_io.py``,
   ``ib2d_reference.py``, ``run_reference.py``) into ``demo_NNN/``;
5. writes the demo ``.gitignore``;
6. prints the remaining to-do list for the human/agent.

Nothing is run automatically: the port of the example-specific ``*.m`` user
modules, the choice of a feasible reduced setting, the verification and the
Chinese ``readme.md`` are left to the caller (they need judgement).
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "_template")
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 3))
DEFAULT_EXAMPLES = os.path.join(REPO, "third_party", "ib2d", "matIB2d",
                               "Examples")

TEMPLATE_FILES = ("main.py", "compare.py", "ib2d_io.py", "ib2d_reference.py",
                  "run_reference.py")
GITIGNORE = "plot*/\nib2d_run*/\nfigures/\n__pycache__/\n*.log\n"
# small text/data files some examples need (auxiliary tables the ported user
# modules read); large binary/media files are deliberately not copied
AUX_SUFFIXES = (".txt", ".dat", ".csv", ".phases", ".session.gui")


def read_string_name(example_dir):
    """``string_name`` of the example's input2d (without importing ib2d_io)."""
    name = None
    with open(os.path.join(example_dir, "input2d")) as fh:
        for line in fh:
            line = line.split("%", 1)[0].strip()
            if line.startswith("string_name"):
                name = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return name or "structure"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("number", help="demo number, e.g. 480 (or demo_480)")
    ap.add_argument("example", help="upstream example directory "
                                    "(relative to the matIB2d Examples dir, or "
                                    "absolute)")
    ap.add_argument("--examples-root", default=DEFAULT_EXAMPLES,
                    help="matIB2d Examples directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing demo directory (dangerous)")
    args = ap.parse_args()

    nnn = args.number[5:] if args.number.startswith("demo_") else args.number
    assert nnn.isdigit(), f"demo number must be numeric, got {args.number!r}"
    demo_name = f"demo_{int(nnn):03d}"
    demo_dir = os.path.join(HERE, demo_name)

    src = args.example
    if not os.path.isdir(src):
        src = os.path.join(args.examples_root, args.example)
    assert os.path.isdir(src), f"upstream example directory not found: {src}"
    assert os.path.exists(os.path.join(src, "input2d")), \
        f"refusing to run: {src} has no input2d file"

    if os.path.exists(demo_dir):
        if not args.force:
            sys.exit(f"refusing to run: {demo_dir} already exists")
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    name = read_string_name(src)
    inp = os.path.join(demo_dir, "ib2d_input")
    os.makedirs(inp)

    copied, skipped = [], []
    for fn in sorted(os.listdir(src)):
        path = os.path.join(src, fn)
        if not os.path.isfile(path):
            continue
        is_data = fn == "input2d" or fn.startswith(name + ".")
        is_module = fn.endswith((".m", ".C", ".mat"))
        is_aux = fn.endswith(AUX_SUFFIXES)
        if is_data or is_module or is_aux:
            # main2d.m / main.m are the MATLAB drivers: kept for reference
            shutil.copy(path, os.path.join(inp, fn))
            copied.append(fn)
        else:
            skipped.append(fn)

    for fn in TEMPLATE_FILES:
        shutil.copy(os.path.join(TEMPLATE, fn), os.path.join(demo_dir, fn))
    with open(os.path.join(demo_dir, ".gitignore"), "w") as fh:
        fh.write(GITIGNORE)

    print(f"created {demo_dir}")
    print(f"  upstream      : {src}")
    print(f"  string_name   : {name}")
    print(f"  ib2d_input/   : {len(copied)} files copied verbatim")
    for fn in copied:
        print(f"      {fn}")
    if skipped:
        print(f"  not copied    : {', '.join(skipped)}")
    print(f"  template code : {', '.join(TEMPLATE_FILES)}")

    print(f"""
TODO (human/agent) -- finish {demo_name}:

  1. Port every example-specific MATLAB user module of ib2d_input/ (*.m) to
     python, keeping the file name the port needs:
       update_Springs.m                    -> update_Springs.py
       update_Damped_Springs.m             -> update_Damped_Springs.py
       update_Target_Point_Positions.m     -> update_Target_Point_Positions.py
       update_Beams.m                      -> update_Beams.py
       update_nonInv_Beams.m               -> update_nonInv_Beams.py
       please_Compute_External_Forcing.m   -> please_Compute_External_Forcing.py
       give_Me_General_User_Defined_Force_Densities.m -> (same .py)
       give_Muscle_Activation.m            -> give_Muscle_Activation.py
     Keep the signature/return of the .m module; the identifiers arrive 0-based
     (ib2d_io.load_example rebases them) and MATRIX index columns are 0-based
     in numpy (MATLAB col 4 -> python col 3).
  2. Choose a feasible reduced setting (the shipped grids are often
     128^2..512^2 with 1e4..5e5 steps): document NX/NY/DT/TFINAL/PRINT_DUMP
     and use `NX=.. NY=.. DT=.. TFINAL=.. PRINT_DUMP=.. python main.py` and
     `python run_reference.py --nx .. --dt .. --tend .. --print-dump ..`.
     If the example needs grad-div stabilisation, document GRAD_DIV.
  3. Run reference + port + compare and check the numbers:
       python run_reference.py ...      # pyIB2d -> ib2d_reference.npz
       python main.py ...               # AFSI    -> plot/afsi_result.npz
       python compare.py                # figures/ + compare_table.csv
  4. Write readme.md (CHINESE): upstream path, physical setup, porting notes,
     the results table taken from compare.py's own output, an explicit list of
     what is NOT comparable, and the exact commands with measured runtimes.
  5. Delete the upstream main2d.m/main.m if it is of no use, keep the demo
     self-contained, and never modify other demos.""")


if __name__ == "__main__":
    main()
