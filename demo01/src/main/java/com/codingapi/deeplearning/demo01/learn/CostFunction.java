package com.codingapi.deeplearning.demo01.learn;

/**
 * 代价函数
 * @author lorne
 * @date 2019-10-22
 * J(θ0,θ1) = 1/2m ∑...  (sorry，实在打不出来，看图片吧)
 * 见:images/cost&hypothesis.jpeg
 *
 */
public class CostFunction {

    //实际的Y值
    private double[] y;

    //通过假如函数计算出来的Y值 y^
    private double[] yy;

    //假如函数与实际值的初始化
    public CostFunction(double[] y, double[] yy) {
        this.y = y;
        this.yy = yy;
    }

    public double getVal(){
        double sum = 0;

        int number = y.length;

        //计算差值的平方
        for(int i=0;i<number;i++){
            sum+=(yy[i]-y[i])*(yy[i]-y[i]);
        }

        //计算平均值
        return sum/number/2;
    }

}
