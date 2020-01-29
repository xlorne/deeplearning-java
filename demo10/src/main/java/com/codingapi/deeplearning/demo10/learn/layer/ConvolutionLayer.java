package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.activation.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class ConvolutionLayer implements NeuralNetworkLayer {

    private int channels;
    private int[] kernelSizes;
    private int[] strides;
    private int outChannels;
    private Activation activation;

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


    protected ConvolutionLayer(int channels, int[] kernelSizes, int[] strides, int outChannels, Activation activation) {
        this.channels = channels;
        this.kernelSizes = kernelSizes;
        this.strides = strides;
        this.outChannels = outChannels;
        this.activation = activation;
    }

    private static ConvolutionLayerBuilder builder = new ConvolutionLayerBuilder();

    public static ConvolutionLayerBuilder builder(){
        return builder;
    }
}
