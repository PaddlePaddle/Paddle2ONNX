import onnx
import numpy as np
from base_expect import expect

if __name__ == "__main__":
    node = onnx.helper.make_node(
        "Identity",
        inputs=["x"],
        outputs=["y"],
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[data], outputs=[data], name="test_identity")