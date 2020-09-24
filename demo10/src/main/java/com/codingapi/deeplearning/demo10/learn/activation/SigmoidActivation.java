package com.codingapi.deeplearning.demo10.learn.activation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.protobuf.common.collect.Maps;

/**
 *
 * Sigmoid: f(x) = {1 \over {1+e^{-x}}}
 * @author lorne
 */
@Slf4j
public class SigmoidActivation implements Activation {

    private transient SameDiff sameDiff;

    private transient SDVariable inputs;

    public SigmoidActivation() {
        //通过自动微分 SameDiff 计算
        sameDiff = SameDiff.create();
        inputs =  sameDiff.var("inputs");

        SDVariable exp = sameDiff.math.exp(inputs.mul(-1)).add(1);
        sameDiff.math.pow("outputs",exp,-1);
    }

    @Override
    public INDArray activation(INDArray data) {
        inputs.setArray(data);
        INDArray sameDiffOut = sameDiff.output(Maps.newHashMap(),"outputs").get("outputs");
        return sameDiffOut;
    }

    @Override
    public INDArray derivative(INDArray a) {
        sameDiff.execBackwards(null);
        INDArray gradient =  sameDiff.getGradForVariable("inputs").getArr();
        return gradient;
    }

}
