package com.codingapi.deeplearning.demo02.learn;

import java.math.BigDecimal;

/**
 * 线性回归函数
 * @author lorne
 * @date 2019-10-22
 * @description y=a+bx
 */
public class RegressionFunction {

    private BigDecimal a;
    private BigDecimal b;


    public RegressionFunction(double a, double b) {
        this.a = new BigDecimal(a);
        this.b = new BigDecimal(b);
    }

    public BigDecimal getY(BigDecimal x){
        return a.add(x.multiply(b));
    }

}
