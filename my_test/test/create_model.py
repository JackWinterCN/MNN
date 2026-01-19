#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def generate_unary():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 1, 1, -1))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, -1))
    output = gs.Variable(name="output", dtype=np.float32, )

    # node = gs.Node(op="Concat", inputs=[input0, input1], outputs=[output], attrs={"axis": 3})
    # node = gs.Node(op="M_ADD", inputs=[input0, input1], outputs=[output], attrs={"axis": 0})
    # node = gs.Node(op="Add", inputs=[input0, input1], outputs=[output], attrs={"operand": 2})
    node = gs.Node(op="M_UNARY_POW", inputs=[input0], outputs=[output], attrs={"operand": 2})

    # graph = gs.Graph(nodes=[node], inputs=[input0, input1], outputs=[output])
    graph = gs.Graph(nodes=[node], inputs=[input0], outputs=[output])

    model = gs.export_onnx(graph)
    # onnx.save(model, "add_layer.onnx")
    onnx.save(model, "unary_pow_layer.onnx")

def generate_binary_unary():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 1, 1, -1))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, -1))
    binary_output = gs.Variable(name="binary_output", dtype=np.float32)
    unary_output = gs.Variable(name="output", dtype=np.float32)

    binary_node = gs.Node(op="M_ADD", inputs=[input0, input1], outputs=[binary_output], attrs={"axis": 0})
    unary_node = gs.Node(op="M_UNARY_POW", inputs=[binary_output], outputs=[unary_output], attrs={"operand": 2})

    graph = gs.Graph(nodes=[binary_node, unary_node], inputs=[input0, input1], outputs=[unary_output])

    model = gs.export_onnx(graph)
    onnx.save(model, "add_pow_layer.onnx")

def generate_add_square_pow():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 1, 1, -1))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, -1))
    add_ouput = gs.Variable(name="add_ouput", dtype=np.float32)
    square_ouput = gs.Variable(name="square_ouput", dtype=np.float32)
    pow_output = gs.Variable(name="pow_output", dtype=np.float32)

    add_node = gs.Node(op="M_ADD", inputs=[input0, input1], outputs=[add_ouput], attrs={"axis": 0})
    square_node = gs.Node(op="M_UNARY_SQUARE", inputs=[add_ouput], outputs=[square_ouput])
    unary_node = gs.Node(op="M_UNARY_POW", inputs=[square_ouput], outputs=[pow_output], attrs={"operand": 3})

    graph = gs.Graph(nodes=[add_node, square_node,unary_node], inputs=[input0, input1], outputs=[pow_output])

    model = gs.export_onnx(graph)
    onnx.save(model, "add_square_pow.onnx")


def generate_add_sqrt_pow():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 1, 1, -1))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, -1))
    add_ouput = gs.Variable(name="add_ouput", dtype=np.float32)
    square_ouput = gs.Variable(name="square_ouput", dtype=np.float32)
    pow_output = gs.Variable(name="pow_output", dtype=np.float32)

    add_node = gs.Node(op="M_ADD", inputs=[input0, input1], outputs=[add_ouput], attrs={"axis": 0})
    square_node = gs.Node(op="Sqrt", inputs=[add_ouput], outputs=[square_ouput])
    unary_node = gs.Node(op="M_UNARY_POW", inputs=[square_ouput], outputs=[pow_output], attrs={"operand": 4})

    graph = gs.Graph(nodes=[add_node, square_node,unary_node], inputs=[input0, input1], outputs=[pow_output])

    model = gs.export_onnx(graph)
    onnx.save(model, "add_sqrt_pow.onnx")


def main():
    generate_add_sqrt_pow()


if __name__ == '__main__':
    main()
