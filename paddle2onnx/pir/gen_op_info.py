#!/usr/bin/env python3
"""
Generate C++ op info tables from ops.yaml + op_compat.yaml.

Output: pir_op_info_generated.h — header-only C++ lookup tables.
"""

import os, sys, re, argparse
import yaml

def parse_args(args_str):
    """Parse 'args: (Tensor x, Tensor y, int axis=0)' into [(type, name, default), ...]"""
    args_str = args_str.strip()
    if not args_str.startswith('(') or not args_str.endswith(')'):
        return []
    inner = args_str[1:-1].strip()
    if not inner:
        return []
    result = []
    for arg in split_args(inner):
        arg = arg.strip()
        if not arg:
            continue
        parts = arg.rsplit(None, 1)  # split last word (name)
        if len(parts) == 2:
            atype, rest = parts
            if '=' in rest:
                name, default = rest.split('=', 1)
                result.append((atype.strip(), name.strip(), default.strip()))
            else:
                result.append((atype.strip(), rest.strip(), ''))
        else:
            result.append(('unknown', arg, ''))
    return result

def split_args(s):
    """Split comma-separated args respecting nested parens."""
    depth = 0
    current = ''
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if c == ',' and depth == 0:
            yield current
            current = ''
        else:
            current += c
    if current.strip():
        yield current

def parse_outputs(output_str):
    """Parse 'Tensor(out), Tensor(accuracy)' into [(type, name), ...]"""
    result = []
    for part in split_args(output_str):
        part = part.strip()
        m = re.match(r'(\w+)\((\w+)\)', part)
        if m:
            result.append((m.group(1), m.group(2)))
    return result

def is_tensor_type(t):
    return t.lower() in ('tensor', 'tensor[]')

def generate_cpp(ops_yaml_path, compat_yaml_path, output_path, version):
    """Generate C++ lookup tables from YAML files."""
    with open(ops_yaml_path) as f:
        ops_data = yaml.safe_load(f)
    with open(compat_yaml_path) as f:
        compat_data = yaml.safe_load(f)

    # Build op info maps: op_name → (input_names, output_names)
    op_inputs = {}   # op_name → [input_name, ...]
    op_outputs = {}  # op_name → [output_name, ...]

    for item in ops_data or []:
        op_name = item.get('op', '')
        if not op_name:
            continue

        # Parse inputs from 'args' field
        args_str = item.get('args', '')
        parsed_args = parse_args(args_str)
        input_names = [name for (t, name, default) in parsed_args if is_tensor_type(t)]
        # Also include non-tensor if no tensors found (some ops have only non-tensor args)
        if not input_names:
            input_names = [name for (t, name, default) in parsed_args]

        # Parse outputs
        output_str = item.get('output', '')
        parsed_outputs = parse_outputs(output_str)
        output_names = [name for (t, name) in parsed_outputs]

        op_inputs[op_name] = input_names
        op_outputs[op_name] = output_names

    # Build op name mapping: phi_name → {fluid_name, ...}
    # op_compat format: "- op : abs" / "- op : adadelta_ (adadelta)"
    op_name_map = {}  # new_name → set of old_names
    op_arg_map = {}   # op_name → {phi_arg → fluid_arg}

    for item in compat_data or []:
        op_entry = item.get('op', '')
        if not op_entry:
            continue
        # Parse "phi_name (fluid_name)" or just "phi_name"
        m = re.match(r'(\w+)\s*\((\w+)\)', op_entry)
        if m:
            phi_name = m.group(1)
            fluid_name = m.group(2)
        else:
            phi_name = op_entry.strip()
            fluid_name = phi_name

        if phi_name not in op_name_map:
            op_name_map[phi_name] = set()
        op_name_map[phi_name].add(fluid_name)

        # Parse input/output arg mappings
        for section in ('inputs', 'outputs'):
            mapping = item.get(section, {})
            if isinstance(mapping, dict):
                for phi_arg, fluid_arg in mapping.items():
                    key = f"{phi_name}/{section}/{phi_arg}"
                    op_arg_map[key] = fluid_arg

    # --- Generate C++ ---
    lines = []
    lines.append('// Auto-generated from ops.yaml + op_compat.yaml (version %s)' % version)
    lines.append('// DO NOT EDIT — regenerate via gen_op_info.py')
    lines.append('#pragma once')
    lines.append('#include <string>')
    lines.append('#include <unordered_map>')
    lines.append('#include <vector>')
    lines.append('')
    lines.append('namespace paddle2onnx {')
    lines.append('namespace pir {')
    lines.append('')

    # OpNameNormalizer: old_name → new_name
    lines.append('// Op name mappings: fluid/legacy_name → phi/normalized_name')
    lines.append('inline const std::unordered_map<std::string, std::string>&')
    lines.append('GetOpNameMappings_%s() {' % version)
    lines.append('  static const std::unordered_map<std::string, std::string> m = {')
    for phi_name, old_names in sorted(op_name_map.items()):
        for old_name in sorted(old_names):
            if old_name != phi_name:
                lines.append('    {"%s", "%s"},' % (old_name, phi_name))
    lines.append('  };')
    lines.append('  return m;')
    lines.append('}')
    lines.append('')

    # Input name → index for each op
    lines.append('// Input name → positional index')
    lines.append('inline const std::unordered_map<std::string, int>&')
    lines.append('GetOpInputIndices_%s(const std::string& op_name) {' % version)
    lines.append('  static const std::unordered_map<std::string, std::unordered_map<std::string, int>> all = {')
    for op_name, names in sorted(op_inputs.items()):
        if names:
            lines.append('    {"%s", {' % op_name)
            for i, name in enumerate(names):
                lines.append('      {"%s", %d},' % (name, i))
            lines.append('    }},')
    lines.append('  };')
    lines.append('  static const std::unordered_map<std::string, int> empty;')
    lines.append('  auto it = all.find(op_name);')
    lines.append('  return it != all.end() ? it->second : empty;')
    lines.append('}')
    lines.append('')

    # Output name → index
    lines.append('// Output name → positional index')
    lines.append('inline const std::unordered_map<std::string, int>&')
    lines.append('GetOpOutputIndices_%s(const std::string& op_name) {' % version)
    lines.append('  static const std::unordered_map<std::string, std::unordered_map<std::string, int>> all = {')
    for op_name, names in sorted(op_outputs.items()):
        if names:
            lines.append('    {"%s", {' % op_name)
            for i, name in enumerate(names):
                lines.append('      {"%s", %d},' % (name, i))
            lines.append('    }},')
    lines.append('  };')
    lines.append('  static const std::unordered_map<std::string, int> empty;')
    lines.append('  auto it = all.find(op_name);')
    lines.append('  return it != all.end() ? it->second : empty;')
    lines.append('}')
    lines.append('')

    # Op arg name mappings (phi_name → {phi_arg → fluid_arg})
    lines.append('// Op arg name mappings: phi_arg_name → fluid/legacy_arg_name')
    lines.append('inline const std::unordered_map<std::string, std::string>&')
    lines.append('GetOpArgMappings_%s(const std::string& op_name) {' % version)
    lines.append('  static const std::unordered_map<std::string, std::unordered_map<std::string, std::string>> all = {')
    
    # Group by op_name
    arg_map_by_op = {}
    for key, fluid_arg in op_arg_map.items():
        parts = key.split('/', 2)
        if len(parts) == 3:
            opn, section, arg = parts
            if opn not in arg_map_by_op:
                arg_map_by_op[opn] = {}
            arg_map_by_op[opn][arg] = fluid_arg
    
    for op_name, arg_map in sorted(arg_map_by_op.items()):
        lines.append('    {"%s", {' % op_name)
        for phi_arg, fluid_arg in sorted(arg_map.items()):
            lines.append('      {"%s", "%s"},' % (phi_arg, fluid_arg))
        lines.append('    }},')
    lines.append('  };')
    lines.append('  static const std::unordered_map<std::string, std::string> empty;')
    lines.append('  auto it = all.find(op_name);')
    lines.append('  return it != all.end() ? it->second : empty;')
    lines.append('}')
    lines.append('')

    lines.append('}  // namespace pir')
    lines.append('}  // namespace paddle2onnx')
    lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_path}: {len(op_inputs)} ops, {len(op_outputs)} outputs, {len(op_name_map)} name mappings")
    return op_inputs, op_outputs, op_name_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ops_yaml', required=True)
    parser.add_argument('--compat_yaml', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--version', default='3.4')
    args = parser.parse_args()
    generate_cpp(args.ops_yaml, args.compat_yaml, args.output, args.version)
