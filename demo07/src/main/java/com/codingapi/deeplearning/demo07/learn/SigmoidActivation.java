package com.codingapi.deeplearning.demo07.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description {1 \over {1+e^{-\W \times x}}}
 */
@Slf4j
public class SigmoidActivation implements Activation{

    @Override
    public INDArray forward(INDArray x, INDArray w, INDArray b) {
        int length = x.rows();
        //z = w.Tx+b
        INDArray z = x.mmul(w).add(b.broadcast(length, b.columns()));
        //a = sigmoid(z)
        INDArray res = Transforms.sigmoid(z);

        return res;
    }


    @Override
    public INDArray derivative(INDArray a) {
        return a.muli(a.rsub(1));
    }

}
