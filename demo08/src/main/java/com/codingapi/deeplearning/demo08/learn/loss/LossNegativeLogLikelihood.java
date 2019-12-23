package com.codingapi.deeplearning.demo08.learn.loss;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description Loss = -\sum_i y_i log(a_i)
 */
public class LossNegativeLogLikelihood implements LossFunction {


    @Override
    public double score(INDArray predict, INDArray y) {
        // x 误差值
        INDArray res =  y.mul(Transforms.log(predict)).mul(-1).div(predict.rows());
        return Nd4j.sum(res).sumNumber().doubleValue();
    }

    @Override
    public INDArray gradient(INDArray data, INDArray y) {
        //简化完就是预测值减去y
        return data.sub(y);
    }
}
