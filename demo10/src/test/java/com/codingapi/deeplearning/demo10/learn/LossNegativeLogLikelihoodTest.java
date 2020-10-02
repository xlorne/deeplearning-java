package com.codingapi.deeplearning.demo10.learn;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

/**
 * @author lorne
 * @date 2020/9/24
 * @description
 */
public class LossNegativeLogLikelihoodTest {


    public static void main(String[] args) {


        //test data
        INDArray data = Transforms.softmax(Nd4j.create(new double[]{1.0,0.8,0.5}).reshape(1,3));
        INDArray label  =  Transforms.softmax(Nd4j.create(new double[]{1.0,0.8,0.5}).reshape(1,3));
        System.out.println("inputs:");
        System.out.println(Transforms.softmax(data));
        System.out.println(label);
        //same diff
        SameDiff sameDiff = SameDiff.create();

        SDVariable predict =  sameDiff.var("predict");
        SDVariable labels =  sameDiff.placeHolder("labels", DataType.FLOAT);


        sameDiff.loss().softmaxCrossEntropy("outputs",labels,sameDiff.nn().softmax(predict));

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("labels",Transforms.softmax(label));

        sameDiff.getVariable("predict").setArray(Transforms.softmax(data));

        INDArray res =  sameDiff.output(placeholders,"outputs").get("outputs");

        System.out.println("samediff outs:");
        //samediff out :
        System.out.println(res);


        Map<String,INDArray> placeholders1 = new HashMap<>();
        placeholders1.put("labels",Transforms.softmax(label));
        sameDiff.execBackwards(placeholders1);

        INDArray gradient =  sameDiff.getGradForVariable("predict").getArr();
        System.out.println("gradient outs:");
        //samediff gradient:
        System.out.println(gradient);
    }


}
