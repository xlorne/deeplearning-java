package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.core.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * @author lorne
 * @date 2019-11-15
 * @description
 */
public interface NeuralNetworkLayer extends Serializable {

    /**
     * 前向传播
     * @param data
     * @return
     */
    INDArray forward(INDArray data);

    /**
     * 构建网络层
     * @param layer
     * @param index
     */
    void build(NeuralNetworkLayerBuilder layer,int index);


    /**
     * 初始化参数
     */
    int init(int input, double lamdba, double alpha, long seed);

}
