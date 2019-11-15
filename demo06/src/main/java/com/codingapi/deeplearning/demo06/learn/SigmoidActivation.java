package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description {1 \over {1+e^{-\W \times x}}}
 */
public class SigmoidActivation implements Activation{

    @Override
    public INDArray calculation(INDArray x, INDArray w, INDArray b) {
        int length = x.rows();
        //z = w.Tx+b
        INDArray z = x.mmul(w).add(b.broadcast(length, b.columns()));
        //a = sigmoid(z)
        return Transforms.sigmoid(z);
    }
}
