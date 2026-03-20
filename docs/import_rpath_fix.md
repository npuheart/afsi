# Fixing `ImportError: libdolfinx.so` when importing `afsic`

## Problem

After `pip install .`, running `import afsic` in a fresh terminal raised:

```
ImportError: libdolfinx.so.0.11: cannot open shared object file: No such file or directory
```

This happened even though the build succeeded and the fenicsx-env spack
environment was used to compile the package.

## Root cause

`afsic_ext.so` (the compiled nanobind extension) links against the `coupling`
static library, which in turn links against dolfinx.  The linker records
`libdolfinx.so.0.11` as a runtime dependency inside `afsic_ext.so`.

At compile time the spack environment was active and `LD_LIBRARY_PATH` pointed
to the right library directories, so the build succeeded.  
At import time (fresh terminal, no `spack env activate`) `LD_LIBRARY_PATH` was
empty and the dynamic linker could not locate `libdolfinx.so.0.11`.

## Fix — embed RPATH at build time

The solution is to record the **absolute paths** of all linked libraries
directly inside the `.so` file as RPATH entries.  The dynamic linker reads
these embedded paths before consulting `LD_LIBRARY_PATH`, so the extension
loads correctly regardless of the shell environment.

Two cmake variables are set inside the `if(DEFINED ENV{SPACK_ENV})` block in
[`afsic/CMakeLists.txt`](../afsic/CMakeLists.txt):

```cmake
# Embed all linked-library directories as RPATH so the .so files are
# importable without manually setting LD_LIBRARY_PATH after activation.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
```

| Variable | Effect |
|---|---|
| `CMAKE_INSTALL_RPATH_USE_LINK_PATH` | Automatically appends every directory that contains a linked library to the installed target's RPATH |
| `CMAKE_BUILD_WITH_INSTALL_RPATH` | Applies the same RPATH during the build step (needed for editable installs with `-ve`) |

## Verification

After rebuilding you can confirm the paths are embedded:

```bash
pip install --no-build-isolation -ve .
readelf -d $(python3 -c "import afsic.afsic_ext as m; print(m.__file__)") | grep RPATH
```

Expected output contains the spack view lib directory, e.g.:

```
 0x000000000000000f (RPATH) Library rpath: [/home/staff4/pma/spack/var/spack/environments/fenicsx-env/.spack-env/view/lib:...]
```

After this fix `import afsic` works in any terminal without `spack env activate`.
