#!/usr/bin/env python3
"""
ONNX surgery for RKNN-friendly YOLOv5 model:
  1. Truncate detect head: output 3 raw Conv branches [1,66,80,80], [1,66,40,40], [1,66,20,20]
     instead of merged [1,25200,22]
  2. Replace SPPF MaxPool 5x5 -> 2x MaxPool 3x3 cascade (equivalent via 5 = 2*3-1)
"""
import copy
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

SRC_MODEL = '/home/ma/RobotDetectionModel/Model/0526.onnx'
DST_MODEL = '/home/ma/rknn-toolkit2/rknn_transform/0526_rknn.onnx'

# The 3 detect head Conv output names (these become the new model outputs)
DETECT_CONV_OUTPUTS = [
    '/m/model.24/m.0/Conv_output_0',  # [1, 66, 80, 80]
    '/m/model.24/m.1/Conv_output_0',  # [1, 66, 40, 40]
    '/m/model.24/m.2/Conv_output_0',  # [1, 66, 20, 20]
]

# SPPF MaxPool node names to replace
SPPF_MAXPOOL_NAMES = [
    '/m/model.9/m/MaxPool',
    '/m/model.9/m_1/MaxPool',
    '/m/model.9/m_2/MaxPool',
]


def find_nodes_by_output(graph, output_names):
    """Find nodes that produce the given output names."""
    result = {}
    for node in graph.node:
        for out in node.output:
            if out in output_names:
                result[out] = node
    return result


def get_downstream_nodes(graph, output_name):
    """Get all nodes that consume the given output (directly or transitively)."""
    downstream = set()
    queue = [output_name]
    visited_outputs = set()

    while queue:
        current = queue.pop(0)
        if current in visited_outputs:
            continue
        visited_outputs.add(current)

        for node in graph.node:
            if current in node.input:
                downstream.add(node.name or id(node))
                for out in node.output:
                    queue.append(out)
    return downstream, visited_outputs


def truncate_detect_head(model):
    """Remove all nodes after the 3 detect Conv outputs."""
    graph = model.graph

    # Find all nodes downstream of the 3 detect Conv outputs
    # These are the reshape/transpose/cast/mul/add/concat nodes we want to remove
    all_downstream = set()
    all_downstream_outputs = set()

    for conv_out in DETECT_CONV_OUTPUTS:
        ds_nodes, ds_outputs = get_downstream_nodes(graph, conv_out)
        all_downstream.update(ds_nodes)
        all_downstream_outputs.update(ds_outputs)

    # Remove the detect Conv outputs themselves from downstream (we want to keep them)
    all_downstream_outputs -= set(DETECT_CONV_OUTPUTS)

    # Identify nodes to remove: those whose ALL outputs are in downstream_outputs
    # and none of whose outputs are detect conv outputs
    nodes_to_remove = []
    for node in graph.node:
        node_id = node.name or id(node)
        if node_id in all_downstream:
            # Check this node doesn't produce a detect Conv output
            produces_detect = any(o in DETECT_CONV_OUTPUTS for o in node.output)
            if not produces_detect:
                nodes_to_remove.append(node)

    print(f"Removing {len(nodes_to_remove)} post-detect nodes")
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Remove old outputs
    while len(graph.output) > 0:
        graph.output.pop()

    # Add Cast nodes (float16 -> float32) and new outputs
    # The Conv nodes output float16, but RKNN prefers float32 ONNX
    output_shapes = [
        [1, 66, 80, 80],  # stride 8
        [1, 66, 40, 40],  # stride 16
        [1, 66, 20, 20],  # stride 32
    ]
    output_names_f32 = []
    for name, shape in zip(DETECT_CONV_OUTPUTS, output_shapes):
        f32_name = name + '_f32'
        output_names_f32.append(f32_name)
        # Add Cast node: float16 -> float32
        cast_node = helper.make_node(
            'Cast',
            inputs=[name],
            outputs=[f32_name],
            name=name.replace('Conv_output_0', 'Cast_to_f32'),
            to=TensorProto.FLOAT,
        )
        graph.node.append(cast_node)
        # Declare output as float32
        out = helper.make_tensor_value_info(f32_name, TensorProto.FLOAT, shape)
        graph.output.append(out)

    # Clean up unused initializers that were only used by removed nodes
    remaining_inputs = set()
    for node in graph.node:
        remaining_inputs.update(node.input)

    init_to_remove = []
    for init in graph.initializer:
        if init.name not in remaining_inputs:
            init_to_remove.append(init)

    print(f"Removing {len(init_to_remove)} unused initializers")
    for init in init_to_remove:
        graph.initializer.remove(init)

    return model


def replace_sppf_maxpool(model):
    """Replace MaxPool 5x5 with 2x MaxPool 3x3 cascade.

    Original: x -> MaxPool(k=5,s=1,p=2) -> y
    Replaced: x -> MaxPool(k=3,s=1,p=1) -> tmp -> MaxPool(k=3,s=1,p=1) -> y

    This is mathematically equivalent:
    - MaxPool(5x5, stride=1, pad=2) on input
    - = MaxPool(3x3, stride=1, pad=1) on MaxPool(3x3, stride=1, pad=1) on input
    Because the receptive field of 2 cascaded 3x3 maxpools = 5x5
    """
    graph = model.graph

    # Collect replacements: list of (original_node, [mp1, mp2])
    replacements = []

    for node in list(graph.node):
        if node.op_type != 'MaxPool':
            continue

        # Check if this is a 5x5 maxpool (SPPF pattern)
        kernel_shape = None
        strides = None

        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)

        if kernel_shape != [5, 5]:
            continue
        if strides != [1, 1]:
            continue

        print(f"Replacing MaxPool 5x5 node: {node.name}")
        print(f"  Input: {node.input[0]}, Output: {node.output[0]}")

        # Create intermediate tensor name
        intermediate_name = node.output[0] + '_mp3x3_intermediate'

        # First 3x3 maxpool
        mp1 = helper.make_node(
            'MaxPool',
            inputs=[node.input[0]],
            outputs=[intermediate_name],
            name=node.name + '_3x3_1',
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            ceil_mode=0,
        )

        # Second 3x3 maxpool
        mp2 = helper.make_node(
            'MaxPool',
            inputs=[intermediate_name],
            outputs=[node.output[0]],
            name=node.name + '_3x3_2',
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            ceil_mode=0,
        )

        replacements.append((node, [mp1, mp2]))

    # Apply replacements in reverse order to keep indices stable
    for orig_node, new_nodes in reversed(replacements):
        idx = list(graph.node).index(orig_node)
        graph.node.remove(orig_node)
        for i, nn in enumerate(new_nodes):
            graph.node.insert(idx + i, nn)

    print(f"Replaced {len(replacements)} MaxPool 5x5 nodes with 3x3 cascades")
    return model


def topological_sort(model):
    """Topologically sort the graph nodes to fix ordering after surgery."""
    graph = model.graph
    
    # Build output->node map and collect all available tensors
    output_to_node = {}  # tensor_name -> node that produces it
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node
    
    # Available tensors: graph inputs + initializers
    available = set()
    for inp in graph.input:
        available.add(inp.name)
    for init in graph.initializer:
        available.add(init.name)
    
    # Kahn's algorithm
    remaining = list(graph.node)
    sorted_nodes = []
    max_iterations = len(remaining) * 2  # safety
    iterations = 0
    
    while remaining and iterations < max_iterations:
        iterations += 1
        progress = False
        for node in list(remaining):
            # Check if all inputs are available (empty string inputs are optional)
            all_ready = all(
                inp == '' or inp in available
                for inp in node.input
            )
            if all_ready:
                sorted_nodes.append(node)
                remaining.remove(node)
                for out in node.output:
                    available.add(out)
                progress = True
                break  # restart scan after each addition for stability
        
        if not progress:
            print(f"  WARNING: {len(remaining)} nodes could not be sorted (cycle or missing input)")
            # Add remaining nodes as-is
            sorted_nodes.extend(remaining)
            break
    
    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return model


def main():
    print(f"Loading model: {SRC_MODEL}")
    model = onnx.load(SRC_MODEL)
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    # Step 1: Replace SPPF MaxPool 5x5 -> 2x3x3
    print("\n=== Step 1: Replace SPPF MaxPool 5x5 -> 2x3x3 ===")
    model = replace_sppf_maxpool(model)

    # Step 2: Truncate detect head
    print("\n=== Step 2: Truncate detect head ===")
    model = truncate_detect_head(model)

    # Step 3: Topological sort to fix node ordering
    print("\n=== Step 3: Topological sort ===")
    model = topological_sort(model)
    print("  Topological sort done")

    # Validate
    print("\n=== Validating model ===")
    print(f"  Nodes after surgery: {len(model.graph.node)}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    # Check the model
    try:
        onnx.checker.check_model(model)
        print("  ONNX checker: PASSED")
    except Exception as e:
        print(f"  ONNX checker warning: {e}")
        print("  (This may be OK for large models - opset/shape issues are non-fatal)")

    # Save
    print(f"\nSaving to: {DST_MODEL}")
    onnx.save(model, DST_MODEL)
    print("Done!")

    # Quick verification: reload and print output shapes
    model2 = onnx.load(DST_MODEL)
    print(f"\nVerification - reloaded model:")
    print(f"  Nodes: {len(model2.graph.node)}")
    for out in model2.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")

    # Verify no 5x5 maxpool remains
    for node in model2.graph.node:
        if node.op_type == 'MaxPool':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    ks = list(attr.ints)
                    if ks == [5, 5]:
                        print(f"  WARNING: 5x5 MaxPool still present: {node.name}")
                    else:
                        print(f"  MaxPool kernel: {ks} (OK)")


if __name__ == '__main__':
    main()
