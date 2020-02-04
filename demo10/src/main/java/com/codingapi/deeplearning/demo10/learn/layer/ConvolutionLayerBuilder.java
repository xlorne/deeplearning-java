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
    private int[] padding;
    private Activation activation;

    protected ConvolutionLayerBuilder(){

    }

    private void validateParameter(){

        if(outChannels<=0){
            throw new IllegalArgumentException("outChannels mast greater than zero. ");
        }

        if(channels<=0){
            throw new IllegalArgumentException("channels mast greater than zero. ");
        }

        if(kernelSizes==null||kernelSizes.length!=2){
            throw new IllegalArgumentException("kernelSize length mast is 2.");
        }

        if(padding==null||padding.length!=2){
            throw new IllegalArgumentException("padding length mast is 2.");
        }

        if(strides==null||strides.length!=2){
            throw new IllegalArgumentException("strides length mast is 2.");
        }
    }

    public ConvolutionLayer build(){
        validateParameter();
        return new ConvolutionLayer(channels,kernelSizes,strides,outChannels,activation,padding);
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

    public ConvolutionLayerBuilder padding(int[] padding){
        this.padding = padding;
        return this;
    }

    public ConvolutionLayerBuilder activation(Activation activation) {
        this.activation = activation;
        return this;
    }
    
}
