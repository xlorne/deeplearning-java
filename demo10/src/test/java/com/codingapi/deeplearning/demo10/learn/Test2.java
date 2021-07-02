package com.codingapi.deeplearning.demo10.learn;

import lombok.extern.slf4j.Slf4j;
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
 * @date 2020/10/10
 * @description
 */
@Slf4j
public class Test2 {

    private static final int num_epochs = 64;

    private static final int num_inputs = 28;
    private static final int num_outputs = 10;

    private static final int num_hiddens =  28;

    public static void main(String[] args) {

        SameDiff sameDiff = SameDiff.create();

        SDVariable w1Var = sameDiff.var("w1",DataType.FLOAT);
        SDVariable b1Var = sameDiff.var("b1",DataType.FLOAT);

        SDVariable w2Var = sameDiff.var("w2",DataType.FLOAT);
        SDVariable b2Var = sameDiff.var("b2",DataType.FLOAT);

        INDArray w1 =  Nd4j.rand(num_inputs,num_hiddens);   //28 x 64
        INDArray b1 =  Nd4j.ones(1,num_hiddens);    // 1 x 64

        INDArray w2 = Nd4j.rand(num_hiddens,num_outputs);   //64 x 10
        INDArray b2 = Nd4j.ones(1,num_outputs);     // 1 x 10

        INDArray features = Nd4j.rand(28,28);

        SDVariable featuresVar = sameDiff.placeHolder("feature", DataType.FLOAT);

        SDVariable z1 = sameDiff.dot(featuresVar,w1Var).add(b1Var); //28 x 28 dot 28 x 64

        log.info("shape:{}",Transforms.dot(features,w1).add(b1).shape());

        SDVariable a1 = sameDiff.nn.relu(z1,-1);

//        SDVariable out =  a1.mul("out",1);

        SDVariable z2 = sameDiff.dot(a1,w2Var).add(b2Var);
        SDVariable a2 = sameDiff.nn.softmax(z2);


        SDVariable out =  a2.mul("out",1);



        sameDiff.associateArrayWithVariable(w1, w1Var);
        sameDiff.associateArrayWithVariable(b1, b1Var);
        sameDiff.associateArrayWithVariable(w2, w2Var);
        sameDiff.associateArrayWithVariable(b2, b2Var);


        Map<String,INDArray> placeholders = new HashMap<>();
        placeholders.put("feature",features);

        System.out.println(sameDiff.output(placeholders,"out").get("out"));

        Map<String, INDArray> gradients =  sameDiff.calculateGradients(placeholders,"b1","w2");
//
        System.out.println(gradients.get("w1"));
        System.out.println(gradients.get("w2"));

    }
}
