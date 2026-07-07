#!/usr/bin/env python3
"""
Convert FEBioStudio .fsm (VRP format) file to XDMF mesh format.

The .fsm file is FEBioStudio's internal project format which uses a chunk-based
binary serialization with signature "VRP" (0x00505256).

This script extracts the mesh data (nodes and elements) and writes it to
XDMF format compatible with FEniCS/dolfinx.

Usage:
    python fsm2xdmf.py <input.fsm> <output_prefix>
    
    This produces:
        <output_prefix>.h5   - HDF5 data file
        <output_prefix>.xdmf - XDMF mesh file
"""

import struct
import sys
import os
import numpy as np
from pathlib import Path


# FEBio element types (enum FSElementType from FEElement.h)
# NOTE: enum starts at 1 (FE_INVALID_ELEMENT_TYPE = 0)
FE_ELEMENT_TYPES = {
    0:  ("INVALID", 0),
    1:  ("HEX8",   8),
    2:  ("TET4",   4),
    3:  ("PENTA6", 6),
    4:  ("QUAD4",  4),
    5:  ("TRI3",   3),
    6:  ("BEAM2",  2),
    7:  ("HEX20",  20),
    8:  ("QUAD8",  8),
    9:  ("BEAM3",  3),
    10: ("TET10",  10),
    11: ("TRI6",   6),
    12: ("TET15",  15),
    13: ("HEX27",  27),
    14: ("TRI7",   7),
    15: ("QUAD9",  9),
    16: ("PENTA15",15),
    17: ("PYRA5",  5),
    18: ("TET20",  20),
    19: ("TRI10",  10),
    20: ("TET5",   5),
    21: ("PYRA13", 13),
}

# FEBio to meshio/dolfinx cell type mapping
FEBIO_TO_MESHIO = {
    "TET4":   "tetra",
    "HEX8":   "hexahedron",
    "PENTA6": "wedge",
    "PYRA5":  "pyramid",
    "TRI3":   "triangle",
    "QUAD4":  "quad",
    "TET10":  "tetra10",
    "HEX20":  "hexahedron20",
    "HEX27":  "hexahedron27",
    "PENTA15":"wedge15",
    "PYRA13": "pyramid13",
    "TRI6":   "triangle6",
    "QUAD8":  "quad8",
    "QUAD9":  "quad9",
}

# CID constants from FSCore/enum.h
CID_MASTER           = 0x00000000
CID_VERSION          = 0x00000001
CID_PROJECT          = 0x00020000
CID_MODELINFO        = 0x00030000
CID_FEM              = 0x00040000
CID_OBJ_HEADER       = 0x00080003
CID_MESH             = 0x00090000
CID_MESH_HEADER      = 0x00090001
CID_MESH_NODES       = 0x00090004
CID_MESH_ELEMENTS    = 0x00090005
CID_MESH_FACES       = 0x00090006
CID_MESH_EDGES       = 0x00090007
CID_MESH_NODE_SECTION = 0x00090008
CID_MESH_ELEMENT_SECTION = 0x00090009
CID_MESH_FACE_SECTION  = 0x0009000a
CID_MESH_EDGE_SECTION  = 0x0009000b
CID_MESH_STORAGE       = 0x00090016

CID_MESH_NODE_GID    = 0x00090101
CID_MESH_NODE_POSITION = 0x00090102
CID_MESH_NODE_NID    = 0x00090103

CID_MESH_ELEMENT_TYPE = 0x00090201
CID_MESH_ELEMENT_GID  = 0x00090202
CID_MESH_ELEMENT_NODES = 0x00090203
CID_MESH_ELEMENT_FIBER = 0x00090204
CID_MESH_SHELL_THICKNESS = 0x00090205
CID_MESH_ELEMENT_Q_ACTIVE = 0x00090207
CID_MESH_ELEMENT_Q    = 0x00090208
CID_MESH_ELEMENT_EID  = 0x00090209

CID_MESH_FACE_TYPE     = 0x00090301
CID_MESH_FACE_GID      = 0x00090302
CID_MESH_FACE_NODES    = 0x00090303
CID_MESH_EDGE_TYPE     = 0x00090401
CID_MESH_EDGE_GID      = 0x00090402
CID_MESH_EDGE_NODES    = 0x00090403

# Chunk ID description
CHUNK_NAMES = {
    0x00000000: "CID_MASTER",
    0x00000001: "CID_VERSION",
    0x00020000: "CID_PROJECT",
    0x00030000: "CID_MODELINFO",
    0x00040000: "CID_FEM",
    0x00080003: "CID_OBJ_HEADER",
    0x00090000: "CID_MESH",
    0x00090001: "CID_MESH_HEADER",
    0x00090004: "CID_MESH_NODES",
    0x00090005: "CID_MESH_ELEMENTS",
    0x00090008: "CID_MESH_NODE_SECTION",
    0x00090009: "CID_MESH_ELEMENT_SECTION",
    0x0009000a: "CID_MESH_FACE_SECTION",
    0x0009000b: "CID_MESH_EDGE_SECTION",
    0x00090016: "CID_MESH_STORAGE",
    0x00090101: "CID_MESH_NODE_GID",
    0x00090102: "CID_MESH_NODE_POSITION",
    0x00090103: "CID_MESH_NODE_NID",
    0x00090201: "CID_MESH_ELEMENT_TYPE",
    0x00090202: "CID_MESH_ELEMENT_GID",
    0x00090203: "CID_MESH_ELEMENT_NODES",
    0x00090204: "CID_MESH_ELEMENT_FIBER",
    0x00090205: "CID_MESH_SHELL_THICKNESS",
    0x00090207: "CID_MESH_ELEMENT_Q_ACTIVE",
    0x00090208: "CID_MESH_ELEMENT_Q",
    0x00090209: "CID_MESH_ELEMENT_EID",
    0x00090301: "CID_MESH_FACE_TYPE",
    0x00090302: "CID_MESH_FACE_GID",
    0x00090303: "CID_MESH_FACE_NODES",
    0x00090401: "CID_MESH_EDGE_TYPE",
    0x00090402: "CID_MESH_EDGE_GID",
    0x00090403: "CID_MESH_EDGE_NODES",
}


class FSMReader:
    """Reader for FEBioStudio .fsm (VRP) binary files."""
    
    def __init__(self, filepath):
        self.f = open(filepath, 'rb')
        self.filesize = os.path.getsize(filepath)
        self._pos = 0
    
    def close(self):
        self.f.close()
    
    def tell(self):
        return self.f.tell()
    
    def seek(self, pos, whence=0):
        """Seek to position. If whence=1, pos is relative to current position."""
        if whence == 1:
            self.f.seek(pos, 1)
        else:
            self.f.seek(pos)
    
    def read(self, n):
        """Read n bytes."""
        return self.f.read(n)
    
    def read_uint32(self):
        """Read a little-endian uint32."""
        data = self.f.read(4)
        if len(data) < 4:
            raise EOFError(f"Unexpected EOF at offset {self.tell()}")
        return struct.unpack('<I', data)[0]
    
    def read_int32(self):
        """Read a little-endian int32."""
        data = self.f.read(4)
        if len(data) < 4:
            raise EOFError(f"Unexpected EOF at offset {self.tell()}")
        return struct.unpack('<i', data)[0]
    
    def read_double(self):
        """Read a little-endian double."""
        data = self.f.read(8)
        if len(data) < 8:
            raise EOFError(f"Unexpected EOF at offset {self.tell()}")
        return struct.unpack('<d', data)[0]
    
    def read_vec3d(self):
        """Read a vec3d (3 doubles)."""
        return (self.read_double(), self.read_double(), self.read_double())
    
    def read_string(self):
        """Read a string (int32 length + char[length])."""
        length = self.read_int32()
        if length > 0:
            data = self.f.read(length)
            return data.decode('utf-8', errors='replace')
        return ""
    
    def read_chunks(self, end_offset=None, max_depth=20, depth=0):
        """
        Read chunks until end_offset (exclusive).
        Returns a list of (chunk_id, chunk_size, data_offset) tuples.
        """
        chunks = []
        while True:
            if end_offset is not None and self.tell() >= end_offset:
                break
            if self.tell() >= self.filesize:
                break
            
            try:
                cid = self.read_uint32()
                csize = self.read_uint32()
            except (EOFError, struct.error):
                break
            
            data_offset = self.tell()
            
            if csize == 0:
                # End marker
                break
            
            chunks.append((cid, csize, data_offset))
            
            # Skip the data
            self.seek(data_offset + csize)
        
        return chunks


def find_all_chunks_deep(reader, target_cid, start_offset, end_offset, max_depth=8, depth=0):
    """
    Recursively find ALL chunks with the given ID within a range.
    Returns list of (cid, csize, data_offset) tuples (deduplicated by offset).
    """
    seen_offsets = set()
    results = []
    
    def _search(reader, target_cid, start, end, depth):
        reader.seek(start)
        while reader.tell() < end:
            try:
                cid = reader.read_uint32()
                csize = reader.read_uint32()
            except (EOFError, struct.error):
                break
            
            data_start = reader.tell()
            data_end = data_start + csize
            
            if cid == target_cid and data_start not in seen_offsets:
                seen_offsets.add(data_start)
                results.append((cid, csize, data_start))
            elif csize > 8 and depth < max_depth:
                # Only search inside if we haven't found target here
                _search(reader, target_cid, data_start, data_end, depth + 1)
            
            reader.seek(data_end)
    
    _search(reader, target_cid, start_offset, end_offset, depth)
    return results


def find_chunk_deep(reader, target_cid, start_offset, end_offset, max_depth=8, verbose=False, depth=0):
    """
    Recursively search for a chunk with the given ID within a range.
    Returns (cid, csize, data_offset) or None.
    """
    prefix = "  " * depth
    reader.seek(start_offset)
    while reader.tell() < end_offset:
        try:
            cid = reader.read_uint32()
            csize = reader.read_uint32()
        except (EOFError, struct.error):
            break
        
        data_start = reader.tell()
        data_end = data_start + csize
        
        if verbose and depth <= 3:
            name = CHUNK_NAMES.get(cid, "UNKNOWN")
            if csize < 10000:
                print(f"{prefix}0x{cid:08x} ({name}): size={csize}")
            else:
                print(f"{prefix}0x{cid:08x} ({name}): size={csize} (large)")
        
        if cid == target_cid:
            return (cid, csize, data_start)
        
        # If this is a container chunk (size > 8), search inside it
        if csize > 8 and depth < max_depth:
            result = find_chunk_deep(reader, target_cid, data_start, data_end, 
                                     max_depth, verbose, depth + 1)
            if result:
                return result
        
        reader.seek(data_end)
    
    return None


def find_gobject_name(reader, mesh_offset):
    """
    Given a CID_MESH data offset, find the parent GObject's name.
    Scans backwards byte-by-byte (chunks may not be 4-byte aligned).
    """
    search_start = max(0, mesh_offset - 100000)  # 100KB should be enough
    search_end = mesh_offset
    
    reader.seek(search_start)
    last_name = None
    
    while reader.tell() < search_end - 12:
        pos = reader.tell()
        maybe_cid = reader.read_uint32()
        
        if maybe_cid == 0x00080002:  # CID_OBJ_NAME
            csize = reader.read_uint32()
            if 3 < csize < 256:
                strlen = reader.read_int32()
                if 0 < strlen < 200 and strlen + 4 == csize:
                    name_bytes = reader.read(strlen)
                    try:
                        name = name_bytes.decode('utf-8', errors='strict')
                        if name.isprintable() and name.strip():
                            last_name = name
                    except:
                        pass
            reader.seek(pos + 1)
        else:
            reader.seek(pos + 1)
    
    return last_name


def extract_mesh_data(reader, mesh_chunk, verbose=False):
    """
    Extract mesh data from a CID_MESH chunk.
    
    Returns:
        nodes: numpy array of shape (num_nodes, 3)
        elements: list of (type_id, type_name, connectivity) tuples
    """
    mesh_offset = mesh_chunk[2]
    mesh_end = mesh_offset + mesh_chunk[1]
    reader.seek(mesh_offset)
    
    nodes = None
    elements = []
    nnodes = 0
    nelems = 0
    found_header = False
    
    while reader.tell() < mesh_end:
        try:
            cid = reader.read_uint32()
            csize = reader.read_uint32()
        except (EOFError, struct.error):
            break
        
        data_start = reader.tell()
        data_end = data_start + csize
        
        # Sanity check: first sub-chunk must be CID_MESH_HEADER
        if not found_header and cid != CID_MESH_HEADER and nnodes == 0:
            # This is probably not a real mesh
            reader.seek(data_end)
            continue
        
        found_header = True
        
        if verbose:
            name = CHUNK_NAMES.get(cid, f"UNKNOWN")
            print(f"  Mesh chunk: 0x{cid:08x} ({name}), size={csize}")
        
        if cid == CID_MESH_HEADER:
            while reader.tell() < data_end:
                h_cid = reader.read_uint32()
                h_csize = reader.read_uint32()
                h_data = reader.read(4)
                count = struct.unpack('<I', h_data)[0]
                if h_cid == CID_MESH_NODES:
                    nnodes = count
                    if verbose:
                        print(f"    Nodes: {nnodes}")
                elif h_cid == CID_MESH_ELEMENTS:
                    nelems = count
                    if verbose:
                        print(f"    Elements: {nelems}")
                elif h_cid == CID_MESH_STORAGE:
                    if verbose:
                        print(f"    Storage format: {count}")
        
        elif cid == CID_MESH_NODE_SECTION:
            pos_list = None
            inner_end = data_end
            while reader.tell() < inner_end:
                n_cid = reader.read_uint32()
                n_csize = reader.read_uint32()
                n_data_start = reader.tell()
                
                if n_cid == CID_MESH_NODE_POSITION:
                    pos_raw = reader.read(n_csize)
                    pos_list = np.frombuffer(pos_raw, dtype=np.float64).reshape(-1, 3)
                else:
                    reader.seek(n_data_start + n_csize)
            
            if pos_list is not None:
                nodes = pos_list
                if verbose:
                    print(f"    Node positions: {nodes.shape}")
        
        elif cid == CID_MESH_ELEMENT_SECTION:
            elem_types = None
            elem_nodes_flat = None
            inner_end = data_end
            while reader.tell() < inner_end:
                e_cid = reader.read_uint32()
                e_csize = reader.read_uint32()
                e_data_start = reader.tell()
                
                if e_cid == CID_MESH_ELEMENT_TYPE:
                    elem_types = np.frombuffer(reader.read(e_csize), dtype=np.int32)
                elif e_cid == CID_MESH_ELEMENT_NODES:
                    elem_nodes_flat = np.frombuffer(reader.read(e_csize), dtype=np.int32)
                else:
                    reader.seek(e_data_start + e_csize)
            
            if elem_types is not None and elem_nodes_flat is not None:
                elements_by_type = {}
                pos = 0
                for i in range(len(elem_types)):
                    etype = elem_types[i]
                    info = FE_ELEMENT_TYPES.get(etype)
                    if info is None:
                        continue
                    name, nn = info
                    if nn > 0 and pos + nn <= len(elem_nodes_flat):
                        connectivity = elem_nodes_flat[pos:pos + nn]
                        if etype not in elements_by_type:
                            elements_by_type[etype] = []
                        elements_by_type[etype].append(connectivity)
                        pos += nn
                
                for etype, conn_list in elements_by_type.items():
                    info = FE_ELEMENT_TYPES.get(etype, ("UNKNOWN", 0))
                    conn_array = np.array(conn_list, dtype=np.int32)
                    elements.append((etype, info[0], conn_array))
                    if verbose:
                        print(f"    Element type {etype} ({info[0]}): {len(conn_list)} elements")
        
        reader.seek(data_end)
    
    return nodes, elements


def extract_mesh_from_fsm(filepath, verbose=False):
    """
    Extract ALL mesh data from a FEBioStudio .fsm file.
    
    Returns:
        list of (name, nodes, elements) tuples, one per mesh found
    """
    reader = FSMReader(filepath)
    
    try:
        # 1. Read signature
        sig = reader.read_uint32()
        if sig != 0x00505256:
            raise ValueError(f"Invalid FSM signature: 0x{sig:08x}, expected 0x00505256 (VRP)")
        if verbose:
            print(f"Signature: VRP (0x{sig:08x})")
        
        # 2. Read master chunk
        master_cid = reader.read_uint32()
        master_size = reader.read_uint32()
        master_end = reader.tell() + master_size
        
        if verbose:
            print(f"Master chunk: ID=0x{master_cid:08x}, size={master_size}")
        
        # 3. Find ALL CID_MESH chunks
        if verbose:
            print(f"\nSearching for all CID_MESH (0x{CID_MESH:08x})...")
        
        all_meshes = find_all_chunks_deep(reader, CID_MESH, reader.tell(), master_end, 
                                          max_depth=10)
        
        if not all_meshes:
            raise ValueError("No CID_MESH chunks found in file")
        
        if verbose:
            print(f"\nFound {len(all_meshes)} mesh(es)")
        
        # 4. Extract each mesh
        results = []
        for i, mesh_chunk in enumerate(all_meshes):
            # Try to find the GObject name
            name = find_gobject_name(reader, mesh_chunk[2])
            if not name:
                name = f"mesh_{i+1}"
            
            if verbose:
                print(f"\n=== Mesh {i+1}: \"{name}\" at offset {mesh_chunk[2]}, size={mesh_chunk[1]} ===")
            
            nodes, elements = extract_mesh_data(reader, mesh_chunk, verbose=verbose)
            
            if nodes is None:
                print(f"Warning: No node data found in mesh '{name}', skipping")
                continue
            
            if not elements:
                print(f"Warning: No element data found in mesh '{name}', skipping")
                continue
            
            results.append((name, nodes, elements))
        
        return results
    
    finally:
        reader.close()


def write_xdmf_meshio(output_prefix, nodes, elements, verbose=False):
    """
    Write mesh to XDMF/HDF5 format using meshio.
    Handles mixed meshes by writing separate XDMF files for 3D and 2D elements.
    """
    import meshio
    
    # Separate 3D and 2D elements
    cells_3d = []
    cells_2d = []
    
    for etype_id, etype_name, connectivity in elements:
        cell_type = FEBIO_TO_MESHIO.get(etype_name)
        if cell_type is None:
            print(f"Warning: Cannot map FEBio type '{etype_name}' to meshio type, skipping {len(connectivity)} elements")
            continue
        
        # Check if this is a 2D or 3D element
        if cell_type in ("quad", "triangle", "quad8", "quad9", "triangle6"):
            cells_2d.append((cell_type, connectivity))
        else:
            cells_3d.append((cell_type, connectivity))
    
    xdmf_paths = []
    
    # Write 3D mesh (tetrahedra, hexahedra, wedges, etc.)
    if cells_3d:
        if len(cells_3d) == 1:
            cell_type, connectivity = cells_3d[0]
            cells_list = [meshio.CellBlock(cell_type, connectivity)]
        else:
            cells_list = [meshio.CellBlock(ct, conn) for ct, conn in cells_3d]
        
        mesh = meshio.Mesh(points=nodes, cells=cells_list)
        xdmf_path = f"{output_prefix}.xdmf"
        meshio.write(xdmf_path, mesh)
        xdmf_paths.append(xdmf_path)
        
        if verbose:
            print(f"Written 3D mesh: {xdmf_path}")
            print(f"  Nodes: {len(nodes)}")
            for cell_type, conn in cells_3d:
                print(f"  {cell_type}: {len(conn)} elements")
    
    # Write 2D mesh (shells, quad, triangle)
    if cells_2d:
        if len(cells_2d) == 1:
            cell_type, connectivity = cells_2d[0]
            cells_list = [meshio.CellBlock(cell_type, connectivity)]
        else:
            cells_list = [meshio.CellBlock(ct, conn) for ct, conn in cells_2d]
        
        mesh_2d = meshio.Mesh(points=nodes, cells=cells_list)
        xdmf_path = f"{output_prefix}_surfaces.xdmf"
        meshio.write(xdmf_path, mesh_2d)
        xdmf_paths.append(xdmf_path)
        
        if verbose:
            print(f"Written 2D mesh: {xdmf_path}")
            print(f"  Nodes: {len(nodes)}")
            for cell_type, conn in cells_2d:
                print(f"  {cell_type}: {len(conn)} elements")
    
    return xdmf_paths


def write_xdmf_dolfinx(output_prefix, nodes, elements, verbose=False):
    """Write using dolfinx (falls back to meshio)."""
    return write_xdmf_meshio(output_prefix, nodes, elements, verbose)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert FEBioStudio .fsm file to XDMF mesh format"
    )
    parser.add_argument("input", help="Input .fsm file")
    parser.add_argument("output", nargs="?", help="Output prefix (without extension)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dolfinx", action="store_true", help="Use dolfinx for output (default: meshio)")
    parser.add_argument("--mesh", type=int, default=0, help="Extract only specific mesh index (0=all, 1=first, etc.)")
    
    args = parser.parse_args()
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_prefix = os.path.join(os.path.dirname(input_path) or ".", base)
    
    print(f"Reading: {input_path}")
    print(f"Output prefix: {output_prefix}")
    
    # Extract ALL meshes
    mesh_results = extract_mesh_from_fsm(input_path, verbose=args.verbose)
    
    if not mesh_results:
        print("No meshes found in file!")
        sys.exit(1)
    
    # Filter by mesh index if requested
    if args.mesh > 0:
        if args.mesh <= len(mesh_results):
            mesh_results = [mesh_results[args.mesh - 1]]
        else:
            print(f"Error: Mesh index {args.mesh} out of range (found {len(mesh_results)} meshes)")
            sys.exit(1)
    
    # Write each mesh
    for i, (name, nodes, elements) in enumerate(mesh_results):
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        mesh_prefix = f"{output_prefix}_{safe_name}" if len(mesh_results) > 1 else output_prefix
        
        print(f"\nMesh {i+1}: \"{name}\"")
        print(f"  Nodes: {len(nodes)}")
        for etype_id, etype_name, connectivity in elements:
            print(f"  {etype_name} (type {etype_id}): {len(connectivity)} elements")
        
        # Write output
        if args.dolfinx:
            write_xdmf_dolfinx(mesh_prefix, nodes, elements, verbose=args.verbose)
        else:
            write_xdmf_meshio(mesh_prefix, nodes, elements, verbose=args.verbose)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

# conda run -n base timeout 900 python3 /home/Pengfei.Ma@glasgow.ac.uk/afsi/fsm2xdmf.py /home/Pengfei.Ma@glasgow.ac.uk/afsi/5607.fsm /home/Pengfei.Ma@glasgow.ac.uk/afsi/5607 -v 2>&1