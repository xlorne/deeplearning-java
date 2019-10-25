package com.codingapi.deeplearning.demo02.learn;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 假如函数
 * @author lorne
 * @date 2019-10-22
 * @description y = XT*theta
 */
public class HypothesisFunction {

    private INDArray theta;


    /**
     * 假如函数的初始值
     * @param theta 矩阵参数形式
     */
    public HypothesisFunction(INDArray theta) {
        this.theta = theta;
    }

    /**
     *  y^
     */
    public INDArray getY(INDArray x){
        return theta.transpose().mmul(x);
    }

    /**
     * 更新参数
     * @param theta 矩阵参数形式
     */
    public void updateParam(INDArray theta){
        this.theta = theta;
    }

    public String getParams() {
        return theta.toString();
    }

}
