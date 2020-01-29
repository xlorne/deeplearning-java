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

    public DenseLayerBuilder input(int in, int out) {
        this.in = in;
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
