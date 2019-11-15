package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description J(\theta) = log()
 */
public class SoftMaxLossFunction implements LossFunction {


    @Override
    public INDArray score(INDArray predict, INDArray y) {
        //max y
        int columns =  y.max(1).amaxNumber().intValue();
        // x 误差值
        return y.mul(Transforms.log(predict.getColumn(columns)).mul(-1));
    }

    @Override
    public INDArray gradient(INDArray data, INDArray y) {
        //简化完就是预测值减去y
        return data.sub(y);
    }
}
