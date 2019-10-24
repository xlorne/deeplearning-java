package com.codingapi.deeplearning.demo01.learn;

import lombok.extern.slf4j.Slf4j;

import java.math.BigDecimal;

/**
 * 梯度下降算法
 * @author lorne
 * @date 2019-10-22
 */
@Slf4j
public class GradientDescentAlgorithmFunction {

    /**
     * 学习率α
     */
    private BigDecimal alpha;

    /**
     * 假如函数参数a
     */
    private BigDecimal temp0;
    /**
     * 假如函数参数b
     */
    private BigDecimal temp1;

    /**
     * 训练次数
     */
    private int batch;

    private HypothesisFunction  hypothesisFunction;

    public GradientDescentAlgorithmFunction(double alpha, double temp0, double temp1,int batch) {
        this.alpha = new BigDecimal(alpha);
        this.temp0 = new BigDecimal(temp0);
        this.temp1 = new BigDecimal(temp1);
        this.batch = batch;

        hypothesisFunction = new HypothesisFunction(temp0,temp1);
    }

    /**
     * 梯度下降算法
     * @param x
     * @param y
     */
    public void train(double[] x,double[] y){
        for(int i=0;i<batch;i++){
            BigDecimal gradientTemp0 = derivativeFunction(hypothesisFunction,x,y,0);
            temp0 = temp0.subtract(alpha.multiply(gradientTemp0));

            BigDecimal gradientTemp1 = derivativeFunction(hypothesisFunction,x,y,1);
            temp1 = temp1.subtract(alpha.multiply(gradientTemp1));

            hypothesisFunction.updateParam(temp0,temp1);
            log.info("train count {},params:{}",i,hypothesisFunction.getParams());
        }
        log.info("train over params:{}",hypothesisFunction.getParams());
    }

    /**
     * 代价函数的导数 /m
     * 公式见: images/gradientdescent.png
     * 推导过程:https://blog.csdn.net/xiaopan233/article/details/86718372
     * 这里相当于 将 x0 = 1;
     * x0 x1 ,x0是指 y=ax0+bx1,将x0 看成1 x1 还是之前的x
     *
     */
    private BigDecimal derivativeFunction(HypothesisFunction hypothesisFunction,
                                          double[] x, double[] y,
                                          int paramIndex) {
        int m = y.length;
        BigDecimal sum = new BigDecimal(0);
        for(int i=0;i<m;i++){
            //y^ - y 预测值-实际y值
            BigDecimal subVal =  hypothesisFunction.getY(x[i]).subtract(new BigDecimal(y[i]));
            //差值 * x(i) x0=1,x1 = x(i)
            subVal = subVal.multiply(new BigDecimal(paramIndex==0?1:x[i]));
            //求和
            sum = sum.add(subVal);
        }
        //除总条数
        return sum.divide(new BigDecimal(m),5,BigDecimal.ROUND_UP);
    }


}
