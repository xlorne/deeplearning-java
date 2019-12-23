package com.codingapi.deeplearning.demo08.learn.loss;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2019-11-15
 * @description
 */
public interface LossFunction {

    /**
     * 获取损失函数得分
     * @param predict   预测值
     * @param y         实际值
     * @return          得分值
     */
    double score(INDArray predict, INDArray y);

    /**
     * 损失函数导数
     * @param data
     * @param y
     * @return
     */
    INDArray gradient(INDArray data,INDArray y);

}
