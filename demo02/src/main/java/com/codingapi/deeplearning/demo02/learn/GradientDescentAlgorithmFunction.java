package com.codingapi.deeplearning.demo02.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    private double alpha;

    /**
     * 假如函数参数
     */
    private INDArray thetaTemp;

    /**
     * 训练次数
     */
    private int batch;


    public GradientDescentAlgorithmFunction(double alpha,int batch) {
        this.alpha = alpha;
        this.thetaTemp = Nd4j.rand(2,1);
        this.batch = batch;
    }

    /**
     * https://www.cnblogs.com/bonelee/p/8996304.html
     * 基于矩阵的 梯度下降算法
     * @param array
     */
    public void train(INDArray array){
        INDArray x =  array.getColumns(0,1);
        INDArray y = array.getColumn(2);

        double m = y.columns();

        for(int i=0;i<batch;i++){

            INDArray diff = x.mmul(thetaTemp).sub(y);
            INDArray derivative = x.transpose().mmul(diff).div(m);
            //theta 赋值
            thetaTemp = thetaTemp.sub(derivative.mul(alpha));

            log.info("train count {},params:{}",i,thetaTemp);

        }
        log.info("train over params:{}",thetaTemp);
    }




}
