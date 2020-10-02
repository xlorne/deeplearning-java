package com.codingapi.deeplearning.demo10.learn.loss;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * @author lorne
 * @date 2019-11-15
 * @description Loss = -\sum_i y_i log(a_i)
 */
public class LossNegativeLogLikelihood implements LossFunction {


    private transient SameDiff sameDiff;

    public LossNegativeLogLikelihood() {
        sameDiff = SameDiff.create();

        SDVariable predict =  sameDiff.var("predict");
        SDVariable labels =  sameDiff.placeHolder("labels", DataType.FLOAT);

        sameDiff.loss().softmaxCrossEntropy("outputs",labels,predict);
    }


    @Override
    public double score(INDArray predict, INDArray y) {
        Map<String,INDArray> placeholders = new HashMap<>();

        placeholders.put("labels",y);

        sameDiff.getVariable("predict").setArray(predict);

        INDArray res =  sameDiff.output(placeholders,"outputs").get("outputs");

        return res.sumNumber().doubleValue();
    }

    @Override
    public INDArray gradient(INDArray data, INDArray y) {
        //简化完就是预测值减去y
        Map<String,INDArray> placeholders = new HashMap<>();

        placeholders.put("labels",y);

        sameDiff.execBackwards(placeholders);

        INDArray gradient =  sameDiff.getGradForVariable("predict").getArr();

        return gradient;
    }
}
