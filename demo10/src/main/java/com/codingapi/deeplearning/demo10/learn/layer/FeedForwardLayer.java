package com.codingapi.deeplearning.demo10.learn.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public interface FeedForwardLayer extends NeuralNetworkLayer {


    /**
     * 反向传播
     * @param delta
     * @return
     */
    INDArray backprop(INDArray delta);

    /**
     * 权重值
     * @return
     */
    INDArray w();

    /**
     * 预测值
     * @return
     */
    INDArray a();

    /**
     * 更新参数
     */
    void updateParam();


    /**
     * 是否是输出层
     * @return
     */
    boolean isOutLayer();

}
