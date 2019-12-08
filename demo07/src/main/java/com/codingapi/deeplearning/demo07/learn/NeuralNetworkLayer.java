package com.codingapi.deeplearning.demo07.learn;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2019-11-15
 * @description
 */
public interface NeuralNetworkLayer {

    /**
     * 前向传播
     * @param data
     * @return
     */
    INDArray forward(INDArray data);


    /**
     * 反向传播
     * @param delta
     * @return
     */
    INDArray backprop(INDArray delta);


    /**
     * 初始化参数
     */
    void init(double lamdba,double alpha,long seed);

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
     * 构建网络层
     * @param layer
     * @param index
     */
    void build(NeuralNetworkLayerBuilder layer,int index);

    /**
     * 是否是输出层
     * @return
     */
    boolean isOutLayer();
}
