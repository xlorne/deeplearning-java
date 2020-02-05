package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.activation.Activation;
import com.codingapi.deeplearning.demo10.learn.activation.SigmoidActivation;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class DenseLayerBuilder {

    private int in;
    private int out;
    private Activation activation = new SigmoidActivation();
    private boolean isOutLayer = false;

    protected DenseLayerBuilder() {
    }

    /**
     * 请使用@NeuralNetworkBuilder#inputType() 设置输入
     * 使用@DenseLayerBuilder#nOut()设置返回
     * @param in    输入值大小
     * @param out   输出值大小
     * @return  DenseLayerBuilder
     */
    @Deprecated
    public DenseLayerBuilder input(int in, int out) {
        this.in = in;
        this.out = out;
        return this;
    }


    public DenseLayerBuilder nOut(int out) {
        this.out = out;
        return this;
    }


    public DenseLayerBuilder isOutLayer(boolean isOutLayer) {
        this.isOutLayer = isOutLayer;
        return this;
    }

    public DenseLayerBuilder activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    public DenseLayer build() {
        return new DenseLayer(in, out, activation, isOutLayer);
    }
    
}
