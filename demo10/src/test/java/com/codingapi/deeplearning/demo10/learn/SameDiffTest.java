package com.codingapi.deeplearning.demo10.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.collect.Maps;

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

        SDVariable x = sameDiff.var("x");
        SDVariable y = sameDiff.var("y");

        x.add(y.mul(sameDiff.math.pow(x,2))).add("out",y);

        x.setArray(Nd4j.create(new double[]{2,2}));
        y.setArray(Nd4j.create(new double[]{3,3}));

        System.out.println(sameDiff.output(Maps.newHashMap(),"out").get("out"));

        sameDiff.execBackwards(null);

        System.out.println(sameDiff.getGradForVariable("x").getArr());

        System.out.println(x.getGradient().getArr());

        System.out.println(sameDiff.getGradForVariable("y").getArr());

    }
}
