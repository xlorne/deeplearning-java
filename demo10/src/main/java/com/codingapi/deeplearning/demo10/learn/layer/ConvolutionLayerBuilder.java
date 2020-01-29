package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.activation.Activation;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class ConvolutionLayerBuilder {

    private int channels;
    private int[] kernelSizes;
    private int[] strides;
    private int outChannels;
    private Activation activation;

    protected ConvolutionLayerBuilder(){

    }

    public ConvolutionLayer build(){
        return new ConvolutionLayer(channels,kernelSizes,strides,outChannels,activation);
    }

    public ConvolutionLayerBuilder nIn(int channels) {
        this.channels = channels;
        return this;
    }

    public ConvolutionLayerBuilder kernelSize(int ... kernelSizes) {
        this.kernelSizes = kernelSizes;
        return this;
    }

    public ConvolutionLayerBuilder stride(int ... strides) {
        this.strides = strides;
        return this;
    }

    public ConvolutionLayerBuilder nOut(int channels) {
        this.outChannels = channels;
        return this;
    }

    public ConvolutionLayerBuilder activation(Activation activation) {
        this.activation = activation;
        return this;
    }
    
}
