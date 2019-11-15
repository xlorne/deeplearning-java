package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2019-11-15
 * @description 激活函数
 */
public interface Activation {


    /**
     * 激活函数计算
     *
     * @param x  x
     * @param w     W
     * @param b     b
     * @return
     */
    INDArray calculation(INDArray x, INDArray w, INDArray b);


}
