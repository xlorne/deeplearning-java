package com.codingapi.deeplearning.demo10.learn.activation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.protobuf.common.collect.Maps;

/**
 *
 * Soft max function
 * row_maxes is a row vector (max for each row)
 * row_maxes = rowmaxes(input)
 * diff = exp(input - max) / diff.rowSums()
 * Outputs a probability distribution.
 *
 * @see org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax
 *
 * @author lorne
 */
@Slf4j
public class SoftMaxActivation implements Activation {

    private transient SameDiff sameDiff;

    public SoftMaxActivation() {
        sameDiff = SameDiff.create();

        SDVariable inputs =  sameDiff.var("inputs");
        //row max
        SDVariable maxRow = inputs.max(1).reshape(-1,1);
        //row.sub(row_max)
        SDVariable diff = inputs.sub(maxRow);
        //exp
        SDVariable exp = sameDiff.math.exp(diff);
        //sum
        SDVariable sum = exp.sum(1).reshape(-1,1);
        //exp /over sum
        exp.div("outputs",sum);
    }

    @Override
    public INDArray activation(INDArray data) {
        SDVariable inputs = sameDiff.getVariable("inputs");
        inputs.setArray(data);
        INDArray sameDiffOut = sameDiff.output(Maps.newHashMap(),"outputs").get("outputs");
        return sameDiffOut;
    }

    @Override
    public INDArray derivative(INDArray a) {
        //通过自动微分 SameDiff 计算导数
        sameDiff.calculateGradients(null);
        INDArray gradient =  sameDiff.getGradForVariable("inputs").getArr();
        return gradient;
    }
}
