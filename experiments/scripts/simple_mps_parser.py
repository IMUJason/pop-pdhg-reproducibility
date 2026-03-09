#!/usr/bin/env python
"""
Simple MPS parser for benchmark testing.
"""

import re
import gzip
from typing import Dict, List, Tuple
import numpy as np
from scipy import sparse


def parse_mps_file(filepath: str):
    """
    Parse MPS file and return problem data.

    Returns:
        Tuple of (A, b, c, lb, ub, integer_vars, constraint_sense)
    """
    # Handle .gz files
    if filepath.endswith('.gz'):
        f = gzip.open(filepath, 'rt', encoding='ascii', errors='ignore')
    else:
        f = open(filepath, 'r', encoding='ascii', errors='ignore')

    content = f.read()
    f.close()

    lines = content.split('\n')

    # Data structures
    row_names = {}  # name -> index
    col_names = {}  # name -> index
    constraints = []  # list of (name, sense, rhs)
    obj_name = None

    # For building sparse matrix
    A_data = []
    A_rows = []
    A_cols = []

    # Objective coefficients
    c_dict = {}

    # Bounds (default: 0 <= x <= inf)
    lb_dict = {}
    ub_dict = {}

    # Integer variables
    integer_cols = set()

    # Current section
    section = None
    in_integer_block = False

    # Header info
    header_integer_count = 0

    for line in lines:
        orig_line = line
        line = line.strip()
        if not line:
            continue

        # Parse header comments (lines starting with *)
        if line.startswith('*'):
            if '*INTEGER:' in line:
                try:
                    header_integer_count = int(line.split('*INTEGER:')[1].strip())
                except:
                    pass
            continue

        # Section headers
        if line in ['ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'RANGES', 'ENDATA']:
            section = line
            continue

        if section == 'ROWS':
            # Parse row: sense + name
            parts = line.split()
            if len(parts) >= 2:
                sense, name = parts[0], parts[1]
                if sense == 'N':
                    obj_name = name
                else:
                    row_names[name] = len(row_names)
                    constraints.append((name, sense, 0.0))

        elif section == 'COLUMNS':
            parts = line.split()
            if len(parts) < 3:
                continue

            col_name = parts[0]

            # Check for integer markers
            # MPS format: MARKER_NAME  'MARKER'  'INTORG' or 'INTEND'
            if len(parts) >= 3 and parts[1] == "'MARKER'":
                if "'INTORG'" in parts:
                    in_integer_block = True
                elif "'INTEND'" in parts:
                    in_integer_block = False
                continue

            # Add column if not exists
            if col_name not in col_names:
                col_names[col_name] = len(col_names)

            col_idx = col_names[col_name]

            # Track integer variables from markers
            if in_integer_block:
                integer_cols.add(col_name)

            # Parse entries (can have up to 4 entries per line: col, row1, val1, row2, val2)
            # Format: COLNAME    ROWNAME1    VALUE1    ROWNAME2    VALUE2
            i = 1
            while i + 1 < len(parts):
                row_or_obj = parts[i]
                try:
                    value = float(parts[i + 1])
                    if row_or_obj == obj_name:
                        # This is objective coefficient
                        c_dict[col_idx] = value
                    elif row_or_obj in row_names:
                        # This is a constraint coefficient
                        row_idx = row_names[row_or_obj]
                        A_data.append(value)
                        A_rows.append(row_idx)
                        A_cols.append(col_idx)
                except ValueError:
                    pass
                i += 2

        elif section == 'RHS':
            parts = line.split()
            if len(parts) >= 3:
                row_name = parts[1]
                try:
                    value = float(parts[2])
                    if row_name in row_names:
                        row_idx = row_names[row_name]
                        constraints[row_idx] = (constraints[row_idx][0],
                                               constraints[row_idx][1], value)
                except ValueError:
                    pass

        elif section == 'BOUNDS':
            parts = line.split()
            if len(parts) >= 3:
                bound_type = parts[0]
                col_name = parts[2]

                # Add column if not exists
                if col_name not in col_names:
                    col_names[col_name] = len(col_names)
                col_idx = col_names[col_name]

                if bound_type == 'BV':
                    # Binary variable
                    integer_cols.add(col_name)
                    lb_dict[col_idx] = 0.0
                    ub_dict[col_idx] = 1.0
                elif bound_type in ['LO', 'UP', 'FX'] and len(parts) >= 4:
                    try:
                        value = float(parts[3])
                        if bound_type == 'LO':
                            lb_dict[col_idx] = value
                        elif bound_type == 'UP':
                            ub_dict[col_idx] = value
                        elif bound_type == 'FX':
                            lb_dict[col_idx] = value
                            ub_dict[col_idx] = value
                    except ValueError:
                        pass
                elif bound_type == 'FR':
                    # Free variable
                    lb_dict[col_idx] = -1e10
                    ub_dict[col_idx] = 1e10

    # If header says all variables are integer, mark all as integer
    n = len(col_names)
    # Mark all variables as integer if header indicates most are integer
    if header_integer_count > 0 and header_integer_count >= n - 5:  # Allow small tolerance
        integer_cols = set(col_names.keys())

    # Build matrices
    m = len(row_names)

    # Build constraint matrix
    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(m, n))

    # Build b vector
    b = np.array([c[2] for c in constraints])

    # Build c vector
    c = np.zeros(n)
    for idx, val in c_dict.items():
        c[idx] = val

    # Build bounds (default: 0 <= x <= inf)
    lb = np.zeros(n)
    ub = np.full(n, 1e10)

    for idx in range(n):
        if idx in lb_dict:
            lb[idx] = lb_dict[idx]
        if idx in ub_dict:
            ub[idx] = ub_dict[idx]

    # Integer variables
    integer_vars = [col_names[name] for name in col_names if name in integer_cols]

    # Constraint sense: L = <=, G = >=, E = =
    constraint_sense = [c[1] for c in constraints]

    return A, b, c, lb, ub, integer_vars, constraint_sense


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        A, b, c, lb, ub, integer_vars = parse_mps_file(filepath)
        print(f"Variables: {len(c)}")
        print(f"Constraints: {len(b)}")
        print(f"Integer vars: {len(integer_vars)}")
        print(f"Nonzeros: {A.nnz}")
    else:
        print("Usage: python simple_mps_parser.py <file.mps>")
