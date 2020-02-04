package com.codingapi.deeplearning.demo10.learn.activation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description {1 \over {1+e^{-\W \times x}}}
 */
@Slf4j
public class SigmoidActivation implements Activation {

    @Override
    public INDArray activation(INDArray data) {
        return Transforms.sigmoid(data);
    }

    @Override
    public INDArray derivative(INDArray a) {
        return a.mul(a.rsub(1));
    }

}
