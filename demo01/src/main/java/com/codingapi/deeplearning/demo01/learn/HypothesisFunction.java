package com.codingapi.deeplearning.demo01.learn;

import java.math.BigDecimal;

/**
 * 假如函数
 * @author lorne
 * @date 2019-10-22
 * @description y = a + bx
 */
public class HypothesisFunction {

    private BigDecimal a;
    private BigDecimal b;


    /**
     * 假如函数的初始值
     * @param a
     * @param b
     */
    public HypothesisFunction(double a, double b) {
        this.a = new BigDecimal(a);
        this.b = new BigDecimal(b);
    }

    /**
     *  y^
     */
    public BigDecimal getY(double x){
        return a.add(b.multiply(new BigDecimal(x)));
    }

    /**
     * 更新参数
     * @param a
     * @param b
     */
    public void updateParam(BigDecimal a,BigDecimal b){
        this.a = a;
        this.b = b;
    }

    public String getParams() {
        return String.format("a:%f,b:%f",a.doubleValue(),b.doubleValue());
    }

}
