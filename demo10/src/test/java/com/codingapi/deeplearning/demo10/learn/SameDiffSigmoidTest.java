package com.codingapi.deeplearning.demo10.learn;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.protobuf.common.collect.Maps;

/**
 *
 * @author lorne
 * @date 2020/9/22
 *
 * f(x) = 1 \over {1+e^x}
 * f(x)\prime = {1\over {1+e^{-x}}}\cdot ({ 1 -  {1\over {1+e^{-x}}}})
 */
public class SameDiffSigmoidTest {

    public static void main(String[] args) {

        //手动计算
        INDArray inputs = Nd4j.create(new double[]{-1,0,1});
        System.out.println("inputs:");
        System.out.println(inputs);
        INDArray res = Nd4j.ones(inputs.shape()).divi(Nd4j.ones(inputs.shape()).addi(Transforms.exp(inputs.mul(-1))));
        System.out.println("Transforms:sigmoid:");
        System.out.println(res);

        //通过自动微分 SameDiff 计算
        SameDiff sameDiff = SameDiff.create();

        SDVariable x =  sameDiff.var("inputs");

        SDVariable exp = sameDiff.math.exp(x.mul(-1)).add(1);
        SDVariable outputs = sameDiff.math.pow("outputs",exp,-1);

        x.setArray(inputs);
        INDArray sameDiffOut = sameDiff.output(Maps.newHashMap(),"outputs").get("outputs");
        System.out.println("SameDiff:sigmoid:");
        System.out.println(sameDiffOut);

        //通过自动微分 SameDiff 计算导数
        sameDiff.calculateGradients(null);

        INDArray gradientX =  sameDiff.getGradForVariable("inputs").getArr();
        System.out.println("SameDiff:sigmoid:gradient:");
        System.out.println(gradientX);

    }
}
