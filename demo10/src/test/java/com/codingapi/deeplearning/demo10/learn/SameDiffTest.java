package com.codingapi.deeplearning.demo10.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.collect.Maps;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author lorne
 * @date 2020/9/22
 * https://my.oschina.net/u/1778239/blog/3053929
 *
 * f(x) = x+yx^2+y;
 * {{\partial f} \over {\partial x} }= 1+2xy;
 * {{\partial f} \over {\partial y}} = x^2+1;
 */
@Slf4j
public class SameDiffTest {

    public static void main(String[] args) {

        SameDiff sameDiff = SameDiff.create();

        SDVariable input = sameDiff.placeHolder("input",DataType.FLOAT);

        SDVariable x = sameDiff.var("x", DataType.FLOAT);
        SDVariable y = sameDiff.var("y", DataType.FLOAT);

        SDVariable out =  x.add(y.mul(sameDiff.math.pow(x,2))).add(y);

        out.mul("out",input);

        x.setArray(Nd4j.create(new double[]{2,2}));

        y.setArray(Nd4j.create(new double[]{3,3}));

        Map<String,INDArray> placeholders = new HashMap<>();
        placeholders.put("input",Nd4j.create(new double[]{2,2}));

        System.out.println(sameDiff.output(placeholders,"out").get("out"));

        Map<String, INDArray> gradients =  sameDiff.calculateGradients(placeholders,"x","y");

        System.out.println(gradients.get("x"));
        System.out.println(gradients.get("y"));

    }
}
