package com.codingapi.deeplearning.demo10.learn.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class SubsamplingLayer implements NeuralNetworkLayer {

    private int[] kernelSizes;
    private int[] strides;

    private static  SubsamplingLayerBuilder builder = new SubsamplingLayerBuilder();

    public static SubsamplingLayerBuilder builder() {
        return builder;
    }

    protected SubsamplingLayer(int[] kernelSizes, int[] strides) {
        this.kernelSizes = kernelSizes;
        this.strides = strides;
    }

    @Override
    public INDArray forward(INDArray data) {
        return null;
    }

    @Override
    public INDArray backprop(INDArray delta) {
        return null;
    }

    @Override
    public void init(double lamdba, double alpha, long seed) {

    }

    @Override
    public INDArray w() {
        return null;
    }

    @Override
    public INDArray a() {
        return null;
    }

    @Override
    public void updateParam() {

    }

    @Override
    public void build(NeuralNetworkLayerBuilder layer, int index) {

    }

    @Override
    public boolean isOutLayer() {
        return false;
    }
}
