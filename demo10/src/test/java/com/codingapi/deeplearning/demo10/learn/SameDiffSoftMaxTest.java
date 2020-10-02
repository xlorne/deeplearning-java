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
        INDArray inputs = Nd4j.create(new double[]{ 0.2916,    0.7324,    0.4937,    0.8722,    0.1446,    0.6516,    0.8201,    0.5063,    0.0175,    0.8457 }).reshape(1,10);

        int length = inputs.rows();
        INDArray data = Nd4j.create(inputs.toDoubleMatrix()).reshape(1,10);

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

//        inputs = inputs.add(1);

        INDArray gradientX =  sameDiff.getGradForVariable("inputs").getArr();
        System.out.println("SameDiff:softmax:gradient:");
        System.out.println(gradientX);

        System.out.println("softmax.crossentity:");
        INDArray softmaxcorssentity =  Nd4j.create(new double[]{0.0015,    0.0016,   -0.0141,    0.0016,    0.0015,    0.0016,    0.0016,    0.0016,    0.0015,    0.0016}).reshape(1,10);
        System.out.println(gradientX.mul(softmaxcorssentity));
//         0.0752,    0.1169,   -0.9080,    0.1344,    0.0649,    0.1078,    0.1276,    0.0932,    0.0572,    0.1309


    }


}
