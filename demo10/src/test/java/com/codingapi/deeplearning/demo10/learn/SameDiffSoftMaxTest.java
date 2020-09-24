package com.codingapi.deeplearning.demo10.learn;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.protobuf.common.collect.Maps;

/**
 *
 *
 *
 * f(x) =  e^{i}\over { \sum_j e^i_j}
 * https://blog.csdn.net/bitcarmanlee/article/details/82320853
 *
 * @author lorne
 */
public class SameDiffSoftMaxTest {

    public static void main(String[] args) {

        //手动计算
        INDArray inputs = Nd4j.create(new double[]{0.13,0.34,0.46,1,2,3}).reshape(2,3);

        int length = inputs.rows();
        INDArray data = Nd4j.create(inputs.toDoubleMatrix()).reshape(2,3);

        INDArray maxVal = Nd4j.max(data,1).reshape(length,1);
        INDArray z = data.subi(maxVal);
        INDArray exp = Transforms.exp(z);
        //sum(exp)
        INDArray sum = Nd4j.sum(exp,1).reshape(length,1);
        //exp/sum
        INDArray res =  exp.divi(sum);


        System.out.println("inputs:");
        System.out.println(inputs);

        System.out.println("test:");
        System.out.println(Transforms.softmax(inputs));

        System.out.println("Transforms:softmax:");
        System.out.println(res);

        //通过自动微分 SameDiff 计算
        SameDiff sameDiff = SameDiff.create();

        SDVariable x =  sameDiff.var("inputs");

        SDVariable xmax = x.max(1).reshape(-1,1);

        SDVariable diff = x.sub(xmax);
        SDVariable sexp = sameDiff.math.exp(diff);
        SDVariable ssum = sexp.sum(1).reshape(-1,1);

        sexp.div("outputs",ssum);
        sameDiff.getVariable("inputs").setArray(inputs);

        INDArray sameDiffOut = sameDiff.output(Maps.newHashMap(),"outputs").get("outputs");
        System.out.println("SameDiff:softmax:");
        System.out.println(sameDiffOut);

        //通过自动微分 SameDiff 计算导数
        sameDiff.execBackwards(null);

        INDArray gradientX =  sameDiff.getGradForVariable("inputs").getArr();
        System.out.println("SameDiff:softmax:gradient:");
        System.out.println(gradientX);


    }


}
