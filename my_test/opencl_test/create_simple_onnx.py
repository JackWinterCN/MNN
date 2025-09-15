import numpy as np
import onnx
import onnx_graphsurgeon as gs

def main():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 1, 1, -1))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, -1))
    output = gs.Variable(name="output", dtype=np.float32, )

    node = gs.Node(op="Concat", inputs=[input0, input1], outputs=[output], attrs={"axis": 3})
    # node = gs.Node(op="Add", inputs=[input0, input1], outputs=[output], attrs={"axis": 0})

    graph = gs.Graph(nodes=[node], inputs=[input0, input1], outputs=[output])

    model = gs.export_onnx(graph)
    onnx.save(model, "concat_layer.onnx")    

if __name__ == '__main__':
    main()
