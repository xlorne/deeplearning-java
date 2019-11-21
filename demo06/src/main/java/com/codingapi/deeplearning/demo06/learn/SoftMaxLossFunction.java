package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description Loss = -\sum_i y_i log(a_i)
 */
public class SoftMaxLossFunction implements LossFunction {


    @Override
    public INDArray score(INDArray predict, INDArray y) {
        // x 误差值
        return y.mul(Transforms.log(predict).mul(-1));
    }

    @Override
    public INDArray gradient(INDArray data, INDArray y) {
        //简化完就是预测值减去y
        return data.sub(y);
    }
}
